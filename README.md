# contikipy
Automated Cooja simulation runner and Contiki log parser using python.

## Adding New Plots
1. In the yaml config...
..* Add the name of your plot to 'plot:' in your yaml simulation.
2. In cplogparser.py...
..* Add your plot to 'atomic_function_map' in plot_data() in cplogparser.py.
..* Map your plot name to the plot function name.
..* Add your plot to 'atomic_dict_map' in plot_data in cplogparser.
..* Map your plot name to the required data sets.
..* Write your plot function. It will take in a list of pandas dataframes
    (that you have previously mapped to this function) which you can use for
    generating your plot.
