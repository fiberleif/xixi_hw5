#!/bin/bash

if [ ! -f data.zip ]; then
    wget https://cmu.box.com/shared/static/ebklocte1t2wl4dja61dg1ct285l65md.zip
    mv ebklocte1t2wl4dja61dg1ct285l65md.zip data.zip
fi
if [ ! -f images.zip ]; then
    wget https://cmu.box.com/shared/static/147jyd6uewh1fff96yfh6e11s98v11xm.zip
    mv 147jyd6uewh1fff96yfh6e11s98v11xm.zip images.zip	
fi
unzip data.zip -d ./dataset/nist_data/
unzip images.zip -d ./dataset/nist_images/
rm images.zip
rm data.zip
