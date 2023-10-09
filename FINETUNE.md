# Fine-tuning  Tubelet-contrastive Models


We use the original [SEVERE BENCHMARK](https://github.com/fmthoker/SEVERE-BENCHMARK) codebase for finetuning. We add the **off-the-shelf** scripts to the SEVERE codebase for finetuning tubelet-contrastive models. 
-  Setup the updated [SEVERE BENCHMARK](https://github.com/fmthoker/SEVERE-BENCHMARK) codebase and follow the installation instructions. 
-  For example, to fine-tune Tubelet-Contrast R2plus1d-18 backbone on various downstream domains i.e. **Something-Something V2, Fine-Gym, Diving, UCF101 and HMDB**, you can run the respective scripts in [scripts_domain_shift/scripts_tubelet_contrast](https://github.com/fmthoker/SEVERE-BENCHMARK/tree/main/action_recognition/scripts_domain_shift/scripts_tubelet_contrast) in the action_recognition folder of the SEVERE-BENCHMARK codebase.
-  To fine-tune Tubelet-Contrast R2plus1d-18 backbone for sample sizes i.e. **UCF-1000 and Gym-1000**, you can run the respective scripts in [scripts_sample_sizes/scripts_tubelet_contrast](https://github.com/fmthoker/SEVERE-BENCHMARK/blob/main/action_recognition/scripts_sample_sizes/scripts_tubelet_contrast) in the action_recognition folder of the SEVERE-BENCHMARK codebase.
-  To fine-tune Tubelet-Contrast R2plus1d-18 backbone for finegym actions i.e. **FX-S1 and UB-S1**, you can run the respective scripts in [scripts_finegym_actions/scripts_tubelet_contrast](https://github.com/fmthoker/SEVERE-BENCHMARK/blob/main/action_recognition/scripts_finegym_actions/scripts_tubelet_contrast) in the action_recognition folder of the SEVERE-BENCHMARK codebase.

