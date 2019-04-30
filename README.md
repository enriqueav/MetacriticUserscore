# MetacriticUserscore
Predict the score of metacritic user reviews of videogames.

Using the kaggle dataset at
[https://www.kaggle.com/dahlia25/metacritic-video-game-comments](https://www.kaggle.com/dahlia25/metacritic-video-game-comments)

For a Medium story.
[https://medium.com/@enriqueav/predicting-the-user-score-of-metacritic-user-reviews-of-video-games-using-keras-functional-api-and-87222f7034f6](https://medium.com/@enriqueav/predicting-the-user-score-of-metacritic-user-reviews-of-video-games-using-keras-functional-api-and-87222f7034f6)


### How to run

To execute the notebook in Google Colaboratory
[https://colab.research.google.com/github/enriqueav/MetacriticUserscore/blob/master/metacritic_user_scores.ipynb](https://colab.research.google.com/github/enriqueav/MetacriticUserscore/blob/master/metacritic_user_scores.ipynb)


To run the training locally:

```bash
git clone https://github.com/enriqueav/MetacriticUserscore.git
cd MetacriticUserscore
```

It is recommended to install the dependencies inside a virtualenv:

```bash
virtualenv env --python=python3.6
source env/bin/activate
pip install -r requirements.txt
```

And then, to run train the program as a script

```bash
python trainer.py
```

This will save the trained model as `combined_model.h5`, which then can be loaded to reuse
as 

```python
from tensorflow.python.keras.models import load_model
combined_model = load_model('combined_model.h5')
```