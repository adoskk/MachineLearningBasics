To reproduce the results in this notebook:
* python==3.10
* install requirementst.txt
* download the news headline dataset and put the single csv file under "data" folder: [https://www.kaggle.com/datasets/adammcmurchie/news-headlines-summary-from-select-12-sources](https://www.kaggle.com/datasets/adammcmurchie/news-headlines-summary-from-select-12-sources)
* Download the trained weight file and put it under "model_weights" folder: [https://drive.google.com/file/d/1q6XvP6r658L6XEnvUYY-ChVNNum7vMhV/view?usp=sharing](https://drive.google.com/drive/folders/1--dJ-7tcugnPoN5KallktRsQ7P_Sx63L)


Training/Evaluation loss after training for 10,000 steps on a A100 GPU

![Training Loss](https://github.com/adoskk/MachineLearningBasics/blob/main/NLP/text_generation/images/training_loss.png)

Inference results on 4 testing titles:
![Inference result](https://github.com/adoskk/MachineLearningBasics/blob/main/NLP/text_generation/images/output.png)
