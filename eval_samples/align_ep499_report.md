# Alignment Comparison for EP499 TAXI DRIVER (explicit @ 610s)
- sample_id: `pod_631ec155_610`
- word: `答案`

## Results
- **whispercpp_native**: ok
  - first: 6.64s -> 7.06s, text: 你喜歡這答案嗎
- **mms_char**: error
  - error: Output channels > 65536 not supported at the MPS device. As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be slower than running natively on MPS.
  - occurrences: 0
- **mms_pinyin**: ok
  - first: 6.881987555555555s -> 7.10205111111111s, text: 答案
- **whisperx**: ok
  - first: 6.882s -> 7.142s, text: 答案
- **stable_ts**: ok
  - first: 6.84s -> 7.1s, text: 答案
- **mfa**: skip
  - error: mfa binary not found
  - occurrences: 0