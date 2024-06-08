"""
Sample from a trained model
"""
import re
from pathlib import Path
from contextlib import nullcontext
import torch
from nanogpt.model import GPTConfig, GPT


class NanoGptPlayer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        # -----------------------------------------------------------------------------

        seed = 1337
        # device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
        device = "cpu"
        dtype = "float16"  # 'float32' or 'bfloat16' or 'float16'
        compile = False  # use PyTorch 2.0 to compile the model to be faster
        # -----------------------------------------------------------------------------

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        device_type = (
            "cuda" if "cuda" in device else "cpu"
        )  # for later use in torch.autocast
        ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[dtype]
        ctx = (
            nullcontext()
            if device_type == "cpu"
            else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        )

        # model
        ckpt_path = f"{self.model_name}"
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint["model_args"])
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

        model.eval()
        model.to(device)
        if compile:
            model = torch.compile(model)  # requires PyTorch 2.0 (optional)

        meta = {
            'vocab_size': 32,
            'itos': {
                0: ' ', 1: '#', 2: '+', 3: '-', 4: '.', 5: '0', 6: '1', 7: '2', 8: '3',
                9: '4', 10: '5', 11: '6', 12: '7', 13: '8', 14: '9', 15: ';', 16: '=',
                17: 'B', 18: 'K', 19: 'N', 20: 'O', 21: 'Q', 22: 'R', 23: 'a', 24: 'b',
                25: 'c', 26: 'd', 27: 'e', 28: 'f', 29: 'g', 30: 'h', 31: 'x'
            },
            'stoi': {
                ' ': 0, '#': 1, '+': 2, '-': 3, '.': 4, '0': 5, '1': 6, '2': 7, '3': 8,
                '4': 9, '5': 10, '6': 11, '7': 12, '8': 13, '9': 14, ';': 15, '=': 16,
                'B': 17, 'K': 18, 'N': 19, 'O': 20, 'Q': 21, 'R': 22, 'a': 23, 'b': 24,
                'c': 25, 'd': 26, 'e': 27, 'f': 28, 'g': 29, 'h': 30, 'x': 31
            }
        }
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join([itos[i] for i in l])

        self.encode = encode
        self.decode = decode
        self.model = model
        self.ctx = ctx
        self.device = device

    def get_nanogpt_response(self, game_state: str, temperature: float) -> str:
        num_samples = 1  # number of samples to draw
        top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
        max_new_tokens = 10

        # Remove ["stockfish elo xxx"]\n["stockfish elo xxx"]\n\n from game_state
        # nanogpt was trained only on pgn transcripts
        # game_state = game_state.split("\n\n")[1].strip()

        # Nanogpt was trained on pgn transcripts of this format: 1.e4 e5 2.Nf3 (not 1. e4 e5 2. Nf3)
        # I did this to save on tokens
        # We remove the space after the move number to match the training data
        # game_state = re.sub(r"(\d+\.) ", r"\1", game_state)

        game_state = ";" + game_state
        
        # print('MODEL_INPUT:', game_state)

        start_ids = self.encode(game_state)

        x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]
        with torch.no_grad():
            with self.ctx:
                for k in range(num_samples):
                    y = self.model.generate(
                        x, max_new_tokens, temperature=temperature, top_k=top_k
                    )

                    model_response = self.decode(y[0].tolist())

        model_response = model_response[len(game_state) :]
        if ";" in model_response:
            model_response = model_response.split(";")[0]

        if "+" in model_response:  # UNTESTED
            model_response = model_response.split("+")[0]

        return model_response

    def get_move_from_response(self, response: str) -> str:
        # Parse the response to get only the first move
        moves = response.split()
        if not moves:
            return ""
        first_move = moves[0]

        return first_move

    def get_move(self, game_state: str, temperature: float) -> str:
        completion = self.get_nanogpt_response(game_state, temperature)
        return self.get_move_from_response(completion)

    def get_config(self) -> dict:
        return {"model": self.model_name}
