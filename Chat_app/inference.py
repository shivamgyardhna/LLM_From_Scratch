import torch
import tiktoken
from GPT_Model import * 

class InstructionGPT:

    def __init__(self, cfg, path):
        self.cfg       = cfg
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.model     = self._load(path)

    def _load(self, path):
        model = GPT(self.cfg)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()
        print("Model loaded successfully!")
        return model

    def _format_input(self, entry):
        instruction_text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{entry['instruction']}"
        )
        input_text = (
            f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
        )
        return instruction_text + input_text

    def generate(self, entry, max_new_tokens=10, temp=0.0, topk=None):
        prompt       = self._format_input(entry) + "\n\n### Response:\n"
        input_ids    = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output_ids = generate_text(
                model          = self.model,
                ip_token_id    = input_tensor,
                max_new_tokens = max_new_tokens,
                context_size   = self.cfg["context_len"],
                temp           = temp,
                topk           = topk
            )

        generated_ids = output_ids[0][len(input_ids):]
        response      = self.tokenizer.decode(generated_ids.tolist())

        if "<|endoftext|>" in response:
            response = response.split("<|endoftext|>")[0]

        return response.strip()


def inference():
    i = 0
    # ✅ Fix 1: was "while(exit)" — exit is a built-in function, always True-ish
    # should be "while True" to keep the loop running
    while True:
        if i == 0:
            gpt = InstructionGPT(cfg, path="../Instruction_fine_tune.pth")
            i += 1
        else:
            text = input("Enter the text (or type 'exit' to quit): ")

            if text.lower() == "exit":
                print("Program Exited")
                break
            else:
                # ✅ Fix 2: was gpt.generate(entry) — entry was never defined
                # build the entry dict from the user's input text
                entry = {
                    "instruction": text,
                    "input": ""       # no extra input for simple chat
                }
                response = gpt.generate(entry)
                print(f"\nResponse: {response}\n")


if __name__ == "__main__":

  cfg=GPT_2_config={
    "vocab":50257,
    "context_len":1024,
    "emb_dim":768,
    "n_head":12,
    "n_layer":12,
    "dropout":0.1,
    "qkv_bias":True  
  }


  # ── 2. Load tokenizer ─────────────────────────────────────────────────────────
  tokenizer = tiktoken.get_encoding("gpt2")


  inference()