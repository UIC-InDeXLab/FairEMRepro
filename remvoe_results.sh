
ls | grep "*_results" | xargs -I q rm -rf q # removing results directory 
rm -rf experiments
rm -rf threshold_experiments
