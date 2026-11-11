---
layout: post
title: "Augmenting the Grey Zone: Generative Models for Knee Osteoarthritis"
subtitle: Can synthetic radiographs help distinguish ambiguous early disease grades?
author: Huy Pham
categories: machinelearning
---

Our paper, [*Augmenting the Grey Zone: Generative Models for Ambiguous Kellgren–Lawrence Grade Boundaries in Knee Osteoarthritis*](https://papers.miccai.org/miccai-2026-sat/RHEUM-AI_002.html), was accepted and presented at **RHEUM-AI, a MICCAI 2026 workshop**.

The work was led by **[Ahmed Shamsia](https://www.linkedin.com/in/ahmedsalah882/)**, a master's student supervised by **[Neslihan Bayramoglu](https://www.linkedin.com/in/neslihan-bayramoglu-5433913a0/)**, the paper's last author. I served as Ahmed's co-supervisor. With public dataset and private radiographs provided by **[Simo Saarakkala](https://www.linkedin.com/in/simo-saarakkala/)**, we investigated whether generative models can help classifiers distinguish the early stages of knee osteoarthritis, where the differences in an X-ray can be especially subtle.

## Why focus on the grey zone?

Knee osteoarthritis severity is commonly assessed from radiographs using the 5-grade Kellgren–Lawrence (KL) grading system. Our study focuses on two adjacent-grade tasks: **KL0 vs. KL1** and **KL1 vs. KL2**.

These boundaries are difficult for both readers and machine learning models. The visual changes are small, and the assigned labels can be ambiguous. A model's performance across all grades can hide weaknesses at precisely these transitions.

We asked whether generating additional training images around these difficult boundaries could improve classification. The practical test was whether synthetic images helped on held-out radiographs, including images from independent cohorts.

## What we compared

![Experimental pipeline showing boundary selection, four generative augmentation methods, matched classifier training, and internal and external evaluation]({{ '/assets/images/grey-zone-pipeline.png' | relative_url }})

*The experimental pipeline, from our RHEUM-AI presentation. Click the figure to view it at full size.*

We evaluated four approaches to generating training images:

- **Unconditional GAN:** generates images from a selected set of boundary-relevant examples.
- **Class-conditional GAN:** generates images conditioned on a requested KL grade.
- **Counterfactual GAN:** translates images across a grade boundary, guided by a classifier.
- **Class-conditional diffusion:** synthesizes images using a latent diffusion model with grade conditioning.

The internal experiments used radiographs from the **Osteoarthritis Initiative (OAI)**, with patient-level training, validation, and test splits. We compared five classifier backbones—Siamese, EfficientNet-B5, ConvNeXt, ConvNeXt V2, and MaxViT—alongside two boundary-filter settings and five real-to-synthetic mixing ratios.

Selected models were then evaluated without fine-tuning on two independent cohorts, **CGMH** and **MAKNEE**. This allowed us to examine whether improvements within OAI carried over to a different population and imaging setting.

## What we found

![Best-configuration balanced accuracy on OAI: standard augmentation 0.69 and 0.76, unconditional GAN 0.70 and 0.78, conditional GAN 0.71 and 0.77, counterfactual GAN 0.74 and 0.78, and conditional diffusion 0.71 and 0.75 for KL0 vs. KL1 and KL1 vs. KL2 respectively]({{ '/assets/images/grey-zone-best-results.png' | relative_url }})

*Best configurations on the held-out OAI test set, from the presentation. These are selected configurations rather than averages across all runs; the vertical axis starts at 0.60. Click to enlarge.*

For **KL0 vs. KL1**, the counterfactual GAN produced the strongest internal configurations, improving balanced accuracy by **5–8 percentage points** over standard-augmentation baselines. It was also the only generative strategy to improve average balanced accuracy over standard augmentation for this task across the evaluated configurations.

For **KL1 vs. KL2**, unconditional and counterfactual augmentation were more competitive internally, while conditional diffusion gave the strongest external result on MAKNEE. Across the experiments, a **5:1 real-to-synthetic ratio** and the narrower **35–65% boundary filter** were generally preferable. Adding more synthetic images did not reliably produce better results.

The external evaluation also tempered the internal findings. No strategy transferred consistently across both cohorts. For KL0 vs. KL1, performance on CGMH remained close to chance, including for the real-image baselines. The best method therefore depended on the task, backbone, training mixture, and evaluation cohort.

## What this means—and what remains unresolved

The results support evaluating generative augmentation at specific, difficult decision boundaries. They also show why a strong internal result needs independent evaluation before drawing conclusions about broader usefulness.

There are important limits to what this experiment can establish. Synthetic images were added only to KL1, changing both the training-set size and class balance. We therefore cannot separate the effects of boundary-focused content from those of adding data and rebalancing the classes. Generator quality was monitored using FID rather than expert assessment of anatomical plausibility, and the external cohorts were limited in size.

For me, this project was also an opportunity to co-supervise Ahmed's master's research and help develop a focused question into a comparative study. Congratulations to Ahmed on the workshop paper, and thank you to Neslihan and Simo for the collaboration.

---

Ahmed Shamsia, Ngoc Huy Pham, Simo Saarakkala, and Neslihan Bayramoglu. [*Augmenting the Grey Zone: Generative Models for Ambiguous Kellgren–Lawrence Grade Boundaries in Knee Osteoarthritis*](https://papers.miccai.org/miccai-2026-sat/RHEUM-AI_002.html). Accepted and presented at the MICCAI 2026 Workshop RHEUM-AI.
