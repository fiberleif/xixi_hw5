#!/bin/bash

chmod 777 ./dataset/scripts/*
./dataset/scripts/get_data.sh
./dataset/scripts/fetch_flowers17.sh
./dataset/scripts/fetch_flowers102.sh

