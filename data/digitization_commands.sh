
# WebPlotDigitizer Command Line Usage
# This would be replaced with actual digitization code

# Example for Zhou 2017 Figure 3.2
plotdigitizer zhou2017_fig3.2.png \
    --calibration "0,0,10,0,0,1" \
    --algorithm "averagingWindow" \
    --windowSize 5 \
    --output zhou2017_fig3.2_data.csv

# Example for growth rate curves
plotdigitizer growth_rate_curves.png \
    --xaxis "log" \
    --yaxis "linear" \
    --output growth_rates_digitized.csv
