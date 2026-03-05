from matplotlib import rcParams
from cycler import cycler




rcParams['font.family'] = 'serif'

rcParams['axes.labelsize'] = 11.
rcParams['axes.titlesize'] = 9.
rcParams['xtick.labelsize'] = 9.
rcParams['ytick.labelsize'] = 9.

#rcParams['mathtext.fontset'] = 'dejavuserif'
rcParams['mathtext.fontset'] = 'cm'

rcParams['savefig.dpi'] = 400
rcParams['legend.fontsize'] = 'x-small'

rcParams['lines.linewidth'] = 1.6

#CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
#                  '#f781bf', '#a65628', '#984ea3',
#                  '#999999', '#e41a1c', '#dede00']

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#e41a1c', '#984ea3', '#f781bf',
                  '#999999', '#dede00', '#a65628']

dark2_color_cycle = ['#1b9e77',
		     '#d95f02',
		     '#7570b3',
		     '#e7298a',
		     '#66a61e',
		     '#e6ab02']

dark2_10 = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']
set8 = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf']
david = ['#1f78b4','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666']
fatima = ['#1f78b4','#a6762e','#e31a1c', '#e7298a','#66a61e','#666666','#a6761d','#7570b3']
from collections import OrderedDict
linestyle_tuple = OrderedDict([
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 2))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (4, 4))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 6, 1, 6))),
     ('dashdotted',            (0, (3, 3, 1, 3))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

#styles = ['-', '-.', '--', ':']*2
styles = ['-', '-.', '--', ':', '-', '-', '-', '--']
styles = ['-',
         linestyle_tuple['dotted'],
         linestyle_tuple['densely dotted'],
         linestyle_tuple['densely dashed'],
         linestyle_tuple['dashed'],
         linestyle_tuple['dashdotted'],
         linestyle_tuple['densely dashdotted'],
         linestyle_tuple['densely dashdotdotted']]

#rcParams['axes.prop_cycle'] = cycler('color', CB_color_cycle)

#rcParams['axes.prop_cycle'] = cycler('color', dark2_color_cycle)
#rcParams['axes.prop_cycle'] = cycler('color', dark2_10)
rcParams['axes.prop_cycle'] = (cycler('color', fatima) + cycler('linestyle',styles) )
