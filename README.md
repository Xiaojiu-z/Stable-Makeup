# Stable-Makeup: When Real-World Makeup Transfer Meets Diffusion Model

<a href="https://arxiv.org/abs/2403.07764"><img src="https://img.shields.io/badge/arXiv-2403.07764-b31b1b.svg" height=22.5></a>
![teaser](assets/sm_teaser.jpg)
Our proposed framework, Stable-Makeup, is a novel diffusion-based method for makeup transfer that can robustly transfer a diverse range of real-world makeup styles, from light to extremely heavy makeup.

## Method Details
![method](https://github.com/Xiaojiu-z/Stable-Makeup/blob/main/assets/sm_method.jpg)
Given a source image $\mathit{I_s}$ , a reference makeup image $\mathit{I_m}$ and an obtained facial structure control image $\mathit{I_c}$ , Stable-Makeup utilizes D-P makeup encoder to encode $\mathit{I_m}$. Content and structural encoders are used to encode $\mathit{I_s}$ and $\mathit{I_c}$ respectively. With the aid of the makeup cross-attention layers, Stable-Makeup aligns the facial regions of $\mathit{I_s}$ and $\mathit{I_m}$ , enabling successful transfers the intricate makeup details. After content-structure decoupling training, Stable-Makeup further maintains content and structure of $\mathit{I_s}$ .

## Todo List
1. - [ ] inference and training code
2. - [ ] pre-trained weights

## Citation
```
@misc{zhang2024stablemakeup,
      title={Stable-Makeup: When Real-World Makeup Transfer Meets Diffusion Model}, 
      author={Yuxuan Zhang and Lifu Wei and Qing Zhang and Yiren Song and Jiaming Liu and Huaxia Li and Xu Tang and Yao Hu and Haibo Zhao},
      year={2024},
      eprint={2403.07764},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
