# Project Applied Data Analysis - Group 'JSON DERULO'

This repository holds the project of group **JSON DERULO** for the Applied Data Analysis EPFL cours.

Group members
- Adrián Fernández Lojo - adrian.fernandezlojo@epfl.ch
- Hanna Zwygart - hanna.zwygart@epfl.ch
- Jeremy Alexandre Serillon - jeremy.serillon@epfl.ch
- Mariem Maazoun - mariem.maazoun@epfl.ch
- Marina-Carmen Meredith Romero - marina.meredith@epfl.ch


## Project Structure: where are Milestone P2 and P3?

```
.
├── README.md
├── data/
│   ├── IMPORTANT_README.md
│   ├── newyorker_caption_contest/
│   └── papers
└── src
    ├── Milestone_P2/
    └── Milestone_P3/
```
The content of our milestones can be found as indicated above. The README for each milestone sits in their respective folders.

.

.

.


### Dataset Download URL
[https://drive.google.com/file/d/1Es1XmglK5isHZuLiO2RDQfc1tYrDYzXz/view?usp=sharing](https://drive.google.com/file/d/1Es1XmglK5isHZuLiO2RDQfc1tYrDYzXz/view?usp=sharing)

#### Reference
Zhang, J.*, Jain, L.*, Guo, Y.*, Chen, J., Zhou, K. L., Suresh, S., ... & Nowak, R. (2024). Humor in AI: Massive Scale Crowd-Sourced Preferences and Benchmarks for Cartoon Captioning. arXiv preprint arXiv:2406.10522.

### Relevant papers
- [Humor in AI (PDF)](data/papers/Humor%20in%20AI%20-%20Paper.pdf) --> [URL to article](https://arxiv.org/abs/2406.10522)
- [Do Androids Laugh at Electric Sheep (PDF)](data/papers/Do%20Androids%20Laugh%20at%20Electric%20Sheep%20-%20Paper.pdf) --> [URL to article](https://aclanthology.org/2023.acl-long.41/)

### Other related links
- [https://nextml.github.io/caption-contest-data/](https://nextml.github.io/caption-contest-data/)
- [https://huggingface.co/datasets/jmhessel/newyorker_caption_contest](https://huggingface.co/datasets/jmhessel/newyorker_caption_contest)
 
.

.

.

## VERY IMPORTANT: DATASET PLACEMENT
Download the dataset (see URL below), unzip/extract it, and place the resulting folder inside the ```data/``` directory.
The folder name MUST remain exactly: **```newyorker_caption_contest```**

Do NOT rename this folder. If you change the name, the dataset may get added or saved in the GitHub repo (which is not ideal because the data set is a lot of storage) and code expecting data/newyorker_caption_contest will fail.


Your ```data/``` folder should look like this:
```
data/
├── IMPORTANT_README.md
├── newyorker_caption_contest
│   ├── contests.json
│   ├── data/
│   └── images/
└── papers
    ├── Do Androids Laugh at Electric Sheep - Paper.pdf
    └── Humor in AI - Paper.pdf
```
In VS Code the folder ```newyorker_caption_contest``` should be grayed out showing that git will not track it, like shown below:

![Expected data directory layout](/figures/documentation/data_folder.png)
