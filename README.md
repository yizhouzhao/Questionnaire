# Questionnaire

pip install -r requirements.txt

--------------------------------------------------------------
12/31/2020
# Run survey
~~python run_survey.py --dataset ag_news~~

python run_survey.py --dataset rotten_tomatoes

python run_survey.py --dataset dbpedia_14

python run_survey.py --dataset yelp_polarity

python run_survey.py --dataset yelp_review_full

python run_survey.py --dataset amazon_polarity

python run_survey.py --dataset amazon_us_reviews

----------------------------------------------------------------
1/7/2020
# Run survey for task-sensitive questions

~~python run_survey.py --dataset rotten_tomatoes --questionnaire moviequestions~~

~~python run_survey.py --dataset imdb --questionnaire moviequestions~~

python run_survey.py --dataset ag_news --questionnaire agnewsquestions

python run_survey.py --dataset yelp_polarity --questionnaire restaurantquestions

python run_survey.py --dataset yelp_review_full --questionnaire restaurantquestions

python run_survey.py --dataset amazon_polarity --questionnaire storequestions

python run_survey.py --dataset amazon_us_reviews --questionnaire storequestions
