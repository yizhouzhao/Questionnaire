#!/bin/bash
python run_survey.py --dataset ag_news --questionnaire agnewsquestions --gpu 0 &
python run_survey.py --dataset yelp_polarity --questionnaire restaurantquestions --gpu 1 & 
python run_survey.py --dataset yelp_review_full --questionnaire restaurantquestions --gpu 2 & 
python run_survey.py --dataset amazon_polarity --questionnaire storequestions --gpu 3 & 
python run_survey.py --dataset amazon_us_reviews --questionnaire storequestions --gpu 4