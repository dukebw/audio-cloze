# Alignment Comparison for EP499 TAXI DRIVER (explicit @ 610s)
- sample_id: `pod_631ec155_610`
- word: `答案`

## Results
- **whispercpp_native**: ok
  - first: 6.64s -> 7.06s, text: 你喜歡這答案嗎
- **mms_char**: ok
  - occurrences: 0
- **mms_pinyin**: ok
  - first: 6.881987555555555s -> 7.10205111111111s, text: 答案
- **whisperx**: error
  - error: Weights only load failed. This file can still be loaded, to do so you have two options, [1mdo those steps only if you trust the source of the checkpoint[0m. 
	(1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
	(2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
	WeightsUnpickler error: Unsupported global: GLOBAL pyannote.audio.core.model.Introspection was not an allowed global by default. Please use `torch.serialization.add_safe_globals([pyannote.audio.core.model.Introspection])` or the `torch.serialization.safe_globals([pyannote.audio.core.model.Introspection])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.
  - occurrences: 0
- **stable_ts**: ok
  - first: 6.84s -> 7.1s, text: 答案
- **mfa**: skip
  - error: mfa binary not found
  - occurrences: 0