#! /bin/bash
ls . | grep "_results" | xargs -I q rm -rf q
rm -rf experiments
rm -rf threshold_experiments
