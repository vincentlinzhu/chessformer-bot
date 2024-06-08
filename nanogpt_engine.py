#!/usr/bin/env python3

"""
Some example classes for people who want to create a homemade bot.

With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.
"""

from __future__ import annotations
import os
import re
import random
import chess
from chess.engine import PlayResult
import logging
from homemade import ExampleEngine
from lib.engine_wrapper import FillerEngine, check_for_draw_offer, get_book_move, get_egtb_move, get_online_move, move_time
from nanogpt.nanogpt_module import NanoGptPlayer
import chess.engine
import chess.polyglot
import chess.syzygy
import chess.gaviota
import logging
import datetime
import time
import test_bot.lichess
from lib import config, model, lichess
from lib.timer import Timer, to_seconds
from typing import Any, Optional, Union
OPTIONS_TYPE = dict[str, Any]
MOVE_INFO_TYPE = dict[str, Any]
COMMANDS_TYPE = list[str]
LICHESS_EGTB_MOVE = dict[str, Any]
CHESSDB_EGTB_MOVE = dict[str, Any]
MOVE = Union[chess.engine.PlayResult, list[chess.Move]]
LICHESS_TYPE = Union[lichess.Lichess, test_bot.lichess.Lichess]

from dataclasses import dataclass


@dataclass
class LegalMoveResponse:
    move_san: Optional[str] = None
    move_uci: Optional[chess.Move] = None
    attempts: int = 0
    is_resignation: bool = False
    is_illegal_move: bool = False

# Use this logger variable to print messages to the console or log files.
# logger.info("message") will always print "message" to the console or log file.
# logger.debug("message") will only print "message" if verbose logging is enabled.
logger = logging.getLogger(__name__)


# Return is (move_san, move_uci, attempts, is_resignation, is_illegal_move)
def get_legal_move(
    player: NanoGptPlayer,
    board: chess.Board,
    game_state: str,
    max_attempts: int = 10,
# ) -> LegalMoveResponse:
) -> PlayResult:
    """Request a move from the player and ensure it's legal."""
    move_san = None
    move_uci = None


    for attempt in range(max_attempts):
        move_san = player.get_move(game_state, float(os.environ["TEMPERATURE"]))
        # player.get_move(
        #     board, game_state, min(((attempt / max_attempts) * 1) + 0.001, temperature)
        # )

        # Sometimes when GPT thinks it's the end of the game, it will just output the result
        # Like "1-0". If so, this really isn't an illegal move, so we'll add a check for that.
        if move_san is not None:
            if move_san == "1-0" or move_san == "0-1" or move_san == "1/2-1/2":
                print(f"{move_san}, player has resigned")
                # Does not break the lichess-bot because an empty play result will be noted as a resignation
                # chess.engine.EngineError('Player has resigned')
                resginationMove = PlayResult("", "")
                resginationMove.resigned = True
                return resginationMove #!

        try:
            move_uci = board.parse_san(move_san)
            return PlayResult(move_uci, None)            
            
        except Exception as e:
            print(f"Error parsing move {move_san}: {e}")
            continue

    if move_uci is None:
        move_uci = random.choice(list(board.legal_moves))
        move_san = board.san(random.choice(list(board.legal_moves)))
        return LegalMoveResponse(
            move_san=" " + move_san, move_uci=move_uci, attempts=max_attempts
        )
        # raise chess.engine.EngineError('Failed to find legal move')



class NanoGPTEngine(ExampleEngine):
    def __init__(self, model_name: str, commands: COMMANDS_TYPE, options: OPTIONS_TYPE, stderr: Optional[int],
                 draw_or_resign: config.Configuration, game: Optional[model.Game] = None, name: Optional[str] = None, temperature: float = os.environ["TEMPERATURE"],
                 **popen_args: str):
        self.player = NanoGptPlayer(model_name = model_name)
        self.temperature = temperature
        # super().__init__(options, draw_or_resign) # there are no options or draw_or_resign values in the config.yml file

        self.engine_name = self.__class__.__name__ if name is None else name

        self.engine = FillerEngine(self, name=self.engine_name)

    def search(self, board: chess.Board) -> PlayResult: 
        is_black = board.turn == chess.BLACK
        
        raw_pgn = str(chess.pgn.Game.from_board(board))        
        # > print(raw_pgn)
        # [Event "?"]
        # [Site "?"]
        # [Date "????.??.??"]
        # [Round "?"]
        # [White "?"]
        # [Black "?"]
        # [Result "*"]

        # 1. Nc3 e5 2. e4 Nf6 *
        
        game_state = re.sub(r'\. |\s+', lambda m: '.' if m.group(0) == '. ' else ' ', raw_pgn.split('\n\n')[1]) # this will remove spaces after game number
        # > print(game_state)
        # 1.Nc3 e5 2.e4 Nf6 *
        
        game_state = game_state.replace('*', '') if is_black else game_state.replace('*', f'{board.fullmove_number}.') # removes the *
        # 1.Nc3 e5 2.e4 Nf6 
        return get_legal_move(self.player, board, game_state)
        
    def play_move(self,
                  board: chess.Board,
                  game: model.Game,
                  li: LICHESS_TYPE,
                  setup_timer: Timer,
                  move_overhead: datetime.timedelta,
                  can_ponder: bool,
                  is_correspondence: bool,
                  correspondence_move_time: datetime.timedelta,
                  engine_cfg: config.Configuration,
                  min_time: datetime.timedelta) -> None:
        """
        Play a move.

        :param board: The current position.
        :param game: The game that the bot is playing.
        :param li: Provides communication with lichess.org.
        :param start_time: The time that the bot received the move.
        :param move_overhead: The time it takes to communicate between the engine and lichess.org.
        :param can_ponder: Whether the engine is allowed to ponder.
        :param is_correspondence: Whether this is a correspondence or unlimited game.
        :param correspondence_move_time: The time the engine will think if `is_correspondence` is true.
        :param engine_cfg: Options for external moves (e.g. from an opening book), and for engine resignation and draw offers.
        :param min_time: Minimum time to spend, in seconds.
        :return: The move to play.
        """

        try:
            best_move = self.search(board)
        except chess.engine.EngineError as error:
            BadMove = (chess.IllegalMoveError, chess.InvalidMoveError)
            # if any(isinstance(e, BadMove) for e in error.args):
                # logger.error("Ending game due to bot attempting an illegal move.")
            game_ender = li.abort if game.is_abortable() else li.resign
            game_ender(game.id)
            raise

        # Heed min_time
        elapsed = setup_timer.time_since_reset()
        if elapsed < min_time:
            time.sleep(to_seconds(min_time - elapsed))

        # self.add_comment(best_move, board) # my nanogpt doesn't have a comment function I guess
        # self.print_stats()
        if best_move.resigned and len(board.move_stack) >= 2:
            li.resign(game.id)
        else:
            li.make_move(game.id, best_move)
