#!/bin/bash
python run_survey.py --dataset ag_news --questionnaire agnewsquestions &
python run_survey.py --dataset yelp_polarity --questionnaire restaurantquestions &
python run_survey.py --dataset yelp_review_full --questionnaire restaurantquestions &
python run_survey.py --dataset amazon_polarity --questionnaire storequestions &
python run_survey.py --dataset amazon_us_reviews --questionnaire storequestions & 