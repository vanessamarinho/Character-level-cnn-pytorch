#!/usr/bin/env bash
python train.py --dataset agnews --feature small
python train.py --dataset agnews --feature large
python train.py --dataset dbpedia --feature small
python train.py --dataset dbpedia --feature large
python train.py --dataset yelp_review --feature small
python train.py --dataset yelp_review --feature large
python train.py --dataset yelp_review_polarity --feature small
python train.py --dataset yelp_review_polarity --feature large
python train.py --dataset amazon_review --feature small
python train.py --dataset amazon_review --feature large
python train.py --dataset amazon_polarity --feature small
python train.py --dataset amazon_polarity --feature large
python train.py --dataset sogou_news --feature small
python train.py --dataset sogou_news --feature large
python train.py --dataset yahoo_answers --feature small
python train.py --dataset yahoo_answers --feature large
