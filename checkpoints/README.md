# checkpoints/

Trained module states of the revision that are small enough to host: 25 WinoGrande adapter and scorer states (`winogrande/`) and 6 Qwen3-1.7B-Base path-weight files (`qwen/`). Verify any file with `sha256sum` against the table below.

## WinoGrande (NB-5) — Drive: `MyDrive/FMI/NB5_winogrande/states/`

| File | Bytes | SHA-256 |
|---|---:|---|
| winogrande/head_seed13.pt | 6,021 | cc125c9d666109ef8e5c635ba9df3446dbbdaa3b9c72a389d825e2f1a639aaa0 |
| winogrande/head_seed1676.pt | 6,037 | a908f0c4500b2aa2bc86a3f466e407b12761a5c83e324937ae1958ce6a3e3724 |
| winogrande/head_seed42.pt | 6,021 | e03c35b74bfbc46218106a46e8d1aea177bbe946f997450669102c9544581913 |
| winogrande/head_seed71.pt | 6,021 | f1bea0cf9597f0e8fa2ae2e70dfb65c532b998e372106fd30bf0de17177e6a40 |
| winogrande/head_seed7595.pt | 6,037 | 59e4335e1ad32373f7e1ce3ce8990e657b94cf1f62b0fcccda6e5fd5becbcb2c |
| winogrande/lcca_seed13.pt | 4,210,373 | d8fa6985d5987c300ed7bd10bd7066c5ea5fd3ee5b3e2281d2b6055f674946f7 |
| winogrande/lcca_seed1676.pt | 4,210,403 | 3593830a1cfa21bf28ea4fafebab2bbdfb5c78aefa71e338b935c5b1bd0c78cd |
| winogrande/lcca_seed42.pt | 4,210,373 | ffee0963e5b9abfb6ce11ba194c0be3f723abd44e3c236f01063c92ff7d998a0 |
| winogrande/lcca_seed71.pt | 4,210,373 | ef228c5d20a986185abfa534e7b7b3dc44ac68d40f3797be5610833f605fbf5a |
| winogrande/lcca_seed7595.pt | 4,210,403 | cb63ebf69e08b093bfcb6db7c749eb1c2e68751774e3a7f1cc947b2b4dd89826 |
| winogrande/lora_seed13.pt | 4,363,037 | 4bc173f1b5e0ee86ea5de7722066d01d90ff26608f5338aac3e115249b74461a |
| winogrande/lora_seed1676.pt | 4,363,245 | f02a5d231420fa9e2594da1209d4542ab07330e1606de1525d79027df923c018 |
| winogrande/lora_seed42.pt | 4,363,037 | 3e9e598b9ce3a72583aabfc49d384ae18b06b0e9911ffd435c165e2b0b198b39 |
| winogrande/lora_seed71.pt | 4,363,037 | 3d72055a2896e1d496e43f05dab3c5c97406d6708e35258436df18fd4ef2c88e |
| winogrande/lora_seed7595.pt | 4,363,245 | 80aca12649811e344234c80c0a848fb4906abf10b6628838f16b10109b2da11c |
| winogrande/mlp_seed13.pt | 4,209,848 | 929b14abeb57a6694ea0d46bbc00518b48edc94b8425ffa70a084f9efd541293 |
| winogrande/mlp_seed1676.pt | 4,209,874 | 4dd735ee81454c31edf7e93f2a1b4069a76a05a0d5308e377874f17ed47ad6b1 |
| winogrande/mlp_seed42.pt | 4,209,848 | 4ffb1d8d9d7f7911dac190a9a4c3ab02de59a866979552e81360e8805cdb3ae8 |
| winogrande/mlp_seed71.pt | 4,209,848 | 34ff54fd2ba70946557b8f0c84171c2eff87cbd51c04f0fb5fa57c22a98de367 |
| winogrande/mlp_seed7595.pt | 4,209,874 | 0891da410b498cf303a7a88d4b114d0e4deb0996da30e9fc09fd81ede8620e03 |
| winogrande/token_mixer_seed13.pt | 4,210,215 | 38aef79128b4aa05b6b21188d9379a7d7f7aab8e891a5155a2fac1e0bbe80b78 |
| winogrande/token_mixer_seed1676.pt | 4,210,243 | b175b3a4ba1a1135c88cbcf849688969521b28d38fbfd5d95704d88d27bba38a |
| winogrande/token_mixer_seed42.pt | 4,210,215 | 97d304f3365d7c377a18d55da34228322d8b738eba7a2c5b51455d280480d7e5 |
| winogrande/token_mixer_seed71.pt | 4,210,215 | 0d0d8d1147527539bdf1fa2c72595bb80572f31aaac570f3b7313d81613ffa3a |
| winogrande/token_mixer_seed7595.pt | 4,210,243 | 86ac6fe30c47b7bbdcb1343547150f498c1d3a1e540a5bb34b0f8719d5e334c3 |

Full-fine-tuning states (`fullft_lraa_seed*.pt`, `fullft_none_seed*.pt`, about 1.7 GB each) are not redistributed; their hashes:

| File | Bytes | SHA-256 |
|---|---:|---|
| fullft_lraa_seed13.pt | 1,740,390,728 | 02ed6872792fffc61dbcfacc09ab8af6d68466f79b3a084b9a5b7fd78e764447 |
| fullft_lraa_seed1676.pt | 1,740,391,538 | 4ec37bf2e5116658fbe6ddf64d6c690d455bfb8a154842f22a51aace6470ccf3 |
| fullft_lraa_seed42.pt | 1,740,390,728 | 610797ef19e614b2b2f03e69b53e61a428ba6b01d20cc0fc0439ee6eee76187f |
| fullft_lraa_seed71.pt | 1,740,390,728 | 635aafe8ed5b302c17fe4ba31ce1b2f98ff28d1c5a46b61e0feaf36b4d2f1b07 |
| fullft_lraa_seed7595.pt | 1,740,391,538 | 86215be9752467f2e465d64eabce154cf004f5923156ea004b06357a52ba9519 |
| fullft_none_seed13.pt | 1,736,186,057 | 705e86f175e24f02ca0eb0fbd316ff4d929f7719c0bb95fbdacb27b5b641d3fd |
| fullft_none_seed1676.pt | 1,736,186,853 | 86f97af8fdfd88154c37717b57809753919b4f99fb7f14d5389d0dfff05d3cd9 |
| fullft_none_seed42.pt | 1,736,186,057 | acfbf40700a7a3139ba2b817c1a8dcb2d93b5ef14123a87b5df305c086e4faf5 |
| fullft_none_seed71.pt | 1,736,186,057 | fe5f4d72f09c2ff0de6b859c9844c0d9cde73c953d29b5502677166b7b0eb96b |
| fullft_none_seed7595.pt | 1,736,186,853 | ead92681fa24d096339dca8340157ac23488931656d55adf041272f75bc7b23c |

## Qwen3-1.7B-Base path weights (frozen cells) — Drive: `MyDrive/FPDA/<wave>_progress/`

| File | Bytes | SHA-256 |
|---|---:|---|
| qwen/attention_frozen_seed13_best.safetensors | 4,211,252 | d47383e873f7a830b816fdaba9a5731bb7c4638cbeb843ca635da50413c9558c |
| qwen/mixer_frozen_seed13_best.safetensors | 4,211,180 | 0b15dba7c9f75f0c200b4b60f8b57012ba0ea1940a1c078d852cc4e49fa434d7 |
| qwen/attention_frozen_seed42_best.safetensors | 4,211,252 | 3b6516f329d27eadbd993c4f9563be303f5fc853293df87bf5c69fe97c6e27e1 |
| qwen/mixer_frozen_seed42_best.safetensors | 4,211,180 | 456fc913dfccdb637e00e3f48aa2d78a5e2d10654eb20ac4cb5d2d6a1ce02f11 |
| qwen/attention_frozen_seed71_best.safetensors | 4,211,252 | 871ce186224563a598645906b08cf3ed9c849cdfb733f4ea353de7abcf676e9b |
| qwen/mixer_frozen_seed71_best.safetensors | 4,211,180 | 81d944c261bb5dcffe7cb13dd1f7d5152df87a021a1d483eb63d016f92c8d9f6 |

Trainable Qwen states (`best_model_path_weights.pt`, about 6.9 GB each) are not redistributed; hashes are in `manifests/FMI_manifests_and_decision_rules_v1_2.md`, Part B.3.
