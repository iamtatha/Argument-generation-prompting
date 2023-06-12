# Argument-generation-prompting
Project of CS 772 course at IIT Bombay, Spring 2023.

This project is completed as a part of the course Deep Learning for Natural Language Processing (CS 772) at Indian Institute of Technology Bombay in Spring, 2023. The course is offered by the Dept. of Computer Science and Engineering of the institute, instructed by Prof. Pushpak Bhattacharyya. The project objective has been accomplished by a team of three M.Tech first year students.



# Dataset
We used Reddit/Change My View(CMV) Dataset for Argument Generation. This dataset contains 42k training, 6.4k validation and 7.5k test instances.
**Change My View Corpus**: A metadata subset of conversations made in the r/ChangeMyView subreddit between 1 Jan 2013 – 7 May 2015, with information on the delta(success) of a user’s utterance in convincing the poster.
The folder **Data** contains a sample of dataset from the CMV corpus. The actual dataset can be found here: https://drive.google.com/drive/folders/1GrFPj-Xw6atfEj17DOYvKdwUx4cBCYP7?usp=sharing


# Architecture
We tried to replicate the paper: **Zhe Hu, Hou Pong Chan, and Lifu Huang. 2022. MOCHA: A Multi-Task Training Approach for Coherent Text Generation from Cognitive Perspective. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 10324–10334, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.**
![image](https://github.com/iamtatha/Argument-generation-prompting/assets/57251093/5eab886d-1380-4679-8008-9324b15f0b03)

We used T5 model for multi-task learning. The codes are available above.
