"""
This program read in a csv file, then plot graphs.
The csv file will be generate automatically.

The requirement of the csv are
Must have columns named 'module', 'None', 'number'
The "module" is for mouse hover labels
The "None" is for scatter plot default size
The "number" is for the W44-E44 order buttom plot

To start, key in:

bokeh serve --show davis.py

"""

import glob
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models.widgets import PreText, Select
from bokeh.plotting import figure
import numpy as np
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.embed import autoload_static
import sys

#"""
fn = sorted(glob.glob('/Users/Jamie/VelaEWpointingtest/20*.txt'))
text = np.genfromtxt('/Users/Jamie/12092016Desktop/dvHeader.txt', dtype = "string")
for i in range(len(fn)):
    text = np.c_[ text, np.genfromtxt(fn[i], dtype = "string", usecols=(18), delimiter=',') ]
#print text
df = pd.DataFrame(text)

columns = ['module', 'number', 'None']
columns += ['test%.3i'  %i for i in range(1,len(fn)+1)]

#print columns

#df.columns = ['module', 'number', 'None', 'test1', 'test2', 'test3', 'test4', 'test5', 'test6', 'test7', 'test8', 'test9', 'test10', 'test11', 'test12', 'test13' ]

df.columns = columns

#df.columns = ['module', 'number', 'None', 'test1', 'test2', 'test3', 'test4', 'test5', 'test6', 'test7', 'test8', 'test9', 'test10', 'test11', 'test12', 'test13' ]
#"""
df.to_csv('example.csv')
#print df

#"""
#csvfoo = sorted(glob.glob('datavisualizer.csv'))
csvfoo = sorted(glob.glob('example.csv'))
choice = 0
print "=============================================="
print 'The choice is ', csvfoo[choice]
print "=============================================="
df = pd.read_csv('datavisualizer.csv')
df = pd.read_csv(csvfoo[choice])
#"""

#print df


#sys.exit()

DEFAULT_TICKERS = sorted(df.columns)
DEFAULT_TICKERS.remove('number')
DEFAULT_TICKERS.remove('module')

SIZES = list(range(5, 22, 3))


#MouseHoverlabel = [str(x) for x in df.module]


#DEFAULT_TICKERS = ['AAPL', 'GOOG', 'INTC', 'BRCM', 'YHOO']


#data = pd.read_csv('datavisualizer.csv')
#data = data.set_index('number')
#columns = sorted(df.columns)
#print pd.DataFrame({columns[2]: data[columns[2]]})
#print ticker



def nix(val, lst):
    return [x for x in lst if x != val]

#@lru_cache()
def load_ticker(ticker):
    #fname = join(DATA_DIR, 'table_%s.csv' % ticker.lower())
    #fname = join(DATA_DIR, 'table_goog.csv')
    #data = pd.read_csv(fname, header=None, parse_dates=['date'],names=['date', 'foo', 'o', 'h', 'l', 'c', 'v'])
    #data = data.set_index('date')
    #print  pd.DataFrame({ticker: data.c})
    #return pd.DataFrame({ticker: data.c, ticker+'_returns': data.c.diff()})
    #print data[ticker]
    #print pd.DataFrame({ticker: data.c})
    #return pd.DataFrame({ticker: data.c})
    #data = pd.read_csv('datavisualizer.csv')
    data = pd.read_csv(csvfoo[choice])
    #data = df
    #if ticker == 'number':
    #    data = data.set_index('fanbeam')
    #    return pd.DataFrame({ticker: data[ticker],ticker+'_module':data.module})
    #else:
    data = data.set_index('number')
    #print data[ticker]
    #print ticker
    #return data[ticker]
    #print pd.DataFrame({ticker: data[ticker]})
    return pd.DataFrame({ticker: data[ticker], ticker+'_module':data.module})

#@lru_cache()
def get_data(t1, t2, t3):
    df1 = load_ticker(t1)
    df2 = load_ticker(t2)
    df3 = load_ticker(t3)
    if t3 != 'None':
        df3[t3] = (df3[t3]**2.)**0.5
        df3[t3] /= df3[t3].max()/20.
    else: df3[t3] = 7 #the size of scatter of None
    data = pd.concat([df1, df2, df3], axis=1)
    data = data.dropna()
    data['t1'] = df1[t1]
    data['t2'] = df2[t2]
    data['t3'] = df3[t3]
    data['module'] = df1[t1+'_module']
    #data['t2_module'] = data[t2+'_module']
    #print data
    #data['t1_returns'] = data[t1+'_returns']
    #data['t2_returns'] = data[t2+'_returns']
    #print data
    return data

# set up widgets

stats = PreText(text='', width=300)
ticker1 = Select(title = "X-axis",value='%s' % DEFAULT_TICKERS[5], options= DEFAULT_TICKERS )
ticker2 = Select(title = "Y-axis",value='%s' % DEFAULT_TICKERS[4], options= DEFAULT_TICKERS )
ticker3 = Select(title = "Size"  ,value='%s' % DEFAULT_TICKERS[3], options= DEFAULT_TICKERS )
#size   = Select(title =' Size', value='None', options=['None'] + columns)

# set up plots

#source = ColumnDataSource(data=dict(date=[], t1=[], t2=[], t1_returns=[], t2_returns=[]))
source        = ColumnDataSource(data=dict(number=[], t1=[], t2=[], t3=[], module=[] ))
#source_static = ColumnDataSource(data=dict(date=[], t1=[], t2=[], t1_returns=[], t2_returns=[]))
source_static = ColumnDataSource(data=dict(number=[], t1=[], t2=[], t3=[], module=[] ))

hover = HoverTool(tooltips=[("module, %s, %s" % (ticker1.value.title(), ticker2.value.title()),"@module, @t1, @t2")])

hover2 = HoverTool(tooltips=[("module","@module")])

#tools = ['pan,wheel_zoom,xbox_select,reset',hover2]
tools = 'pan,wheel_zoom,xbox_select,reset'


corr = figure(plot_width=700, plot_height=700,
              tools=['pan,wheel_zoom,xwheel_zoom,ywheel_zoom,lasso_select,box_select, xbox_select,reset,save',hover2], active_drag='pan')
#corr.circle('t1_returns', 't2_returns', size=2, source=source,selection_color="orange", alpha=0.6, nonselection_alpha=0.1, selection_alpha=0.4)

#if ticker3.value != 'None':
#    corr.circle('t1', 't2', size='t3', source=source, color = "blue", selection_color="darkorange", alpha=0.6, nonselection_alpha=1., selection_alpha=1., hover_color='white',hover_alpha=0.5)
#else:
#    corr.circle('t1', 't2', size ='t3', source=source, color = "blue", selection_color="darkorange", alpha=0.6, nonselection_alpha=1., selection_alpha=1., hover_color='white',hover_alpha=0.5)
corr.circle('t1', 't2', size ='t3', source=source, color = "blue", selection_color="red", alpha=0.6, nonselection_alpha=0.5, selection_alpha=0.6, hover_color='white',hover_alpha=0.5)


#ts1 = figure(plot_width=900, plot_height=200, tools=tools, x_axis_type='datetime', active_drag="xbox_select")
ts1 = figure(plot_width=1000, plot_height=200, tools=tools, active_drag="xbox_select")
#ts1.line('number', 't1', source=source_static)
ts1.circle('number', 't1', size = 5, source=source_static)
#ts1.circle('number', 't1', size=5, source=source, color=None, selection_color="orange")
ts1.circle('number', 't1', size=5, source=source,color=None, selection_color="red")
ts1.xaxis.visible = False

xaxislabels = []
for i in range(44,0,-4): xaxislabels.append('W%02i' % i)
for i in range(1,44,4): xaxislabels.append('E%02i' % i)
xaxislabels.append('E44')
xaxislabels_xpos=[]
for i in range(1,352+4*4,4*4):
    xaxislabels_xpos.append('%i' % i)
xaxislabels_ypos=[]
for i in range(1,352+4*4,4*4): xaxislabels_ypos.append('-4.')
ts1.text(xaxislabels_xpos, xaxislabels_ypos, text=xaxislabels, text_font_size="8pt", text_align='center')
#ts1.text(xaxislabels_xpos, text=xaxislabels, text_font_size="8pt", text_align='center')


#ts2 = figure(plot_width=900, plot_height=200, tools=tools, x_axis_type='datetime', active_drag="xbox_select")
ts2 = figure(plot_width=1000, plot_height=200, tools=tools, active_drag="xbox_select")
#ts2.x_range = ts1.x_range
#ts2.line('number', 't2', source=source_static)
ts2.circle('number', 't2', size=5, source=source_static)
ts2.circle('number', 't2', size=5, source=source, color=None, selection_color="red")
ts2.xaxis.visible = False
ts2.text(xaxislabels_xpos, xaxislabels_ypos, text=xaxislabels, text_font_size="8pt", text_align='center')
#ts2.text(xaxislabels_xpos, text=xaxislabels, text_font_size="8pt", text_align='center')

# set up callbacks

def ticker1_change(attrname, old, new):
    #ticker2.options = nix(new, DEFAULT_TICKERS)
    #ticker3.options = nix(new, DEFAULT_TICKERS)
    update()

def ticker2_change(attrname, old, new):
    #ticker1.options = nix(new, DEFAULT_TICKERS)
    #ticker3.options = nix(new, DEFAULT_TICKERS)
    update()

def ticker3_change(attrname, old, new):
    #ticker2.options = nix(new, DEFAULT_TICKERS)
    #ticker1.options = nix(new, DEFAULT_TICKERS)
    update()

def update(selected=None):
    t1, t2, t3 = ticker1.value, ticker2.value, ticker3.value
    data = get_data(t1, t2, t3)
    source.data = source.from_df(data[['t1', 't2', 't3', 'module']])
    source_static.data = source.data

    update_stats(data, t1, t2)

    corr.title.text = '%s vs. %s' % (t1, t2)
    corr.xaxis.axis_label = t1
    corr.yaxis.axis_label = t2
    ts1.title.text, ts2.title.text = t1, t2

def update_stats(data, t1, t2):
    #stats.text = str(data[[t1, t2, t1+'_returns', t2+'_returns']].describe())
    stats.text = str(data[[t1, t2]].describe())

ticker1.on_change('value', ticker1_change)
ticker2.on_change('value', ticker2_change)
ticker3.on_change('value', ticker3_change)


def selection_change(attrname, old, new):
    t1, t2, t3 = ticker1.value, ticker2.value, ticker3.value
    data = get_data(t1, t2, t3)
    selected = source.selected['1d']['indices']
    #print selected
    if selected:
        data = data.iloc[selected, :]
    update_stats(data, t1, t2)

source.on_change('selected', selection_change)

# set up layout
widgets = column(ticker1, ticker2, ticker3, stats)
main_row = row(corr, widgets)
series = column(ts1, ts2)
layout = column(main_row, series)
#layout = column(main_row)

# initialize
update()
curdoc().add_root(layout)
curdoc().title = "DataVisual"
#output_file('jamie', title='Bokeh Plot', autosave=False, mode='cdn', root_dir=None)
#save()
#js, tag = autoload_static(layout, CDN, "./")
#html = file_html(curdoc, CDN, "jamie_plot")