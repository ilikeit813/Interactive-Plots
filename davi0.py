import pandas as pd
import sys
from bokeh.layouts import row, widgetbox, column
from bokeh.models import Select, HoverTool#, SaveTool, LassoSelectTool
from bokeh.palettes import Greys5,Greys4, Greys3#, Greys2
from bokeh.plotting import curdoc, figure, ColumnDataSource
from bokeh.sampledata.autompg import autompg
import numpy as np

df = pd.read_csv('datavisualizer.csv')
#df = pd.read_csv('all_candidates.dat')
#df = pd.read_csv('a.csv')

#print df
#sys.exit()

SIZES = list(range(5, 22, 3)) #scatter radius range

df.MouseHoverlabel = [str(x) for x in df.module]
#df.MouseHoverlabel2 = [str(x) for x in df.w]

columns = sorted(df.columns)
discrete = [x for x in columns if df[x].dtype == object]
#continuous = [x for x in columns if x not in discrete]
#quantileable = [x for x in continuous if len(df[x].unique()) > 20]
#quantileable = [x for x in continuous if len(df[x].unique()) > 1]


def create_figure():
    xs = df[x.value].values
    ys = df[y.value].values


    x_title = x.value.title()
    y_title = y.value.title()
    y2_title = y2.value.title()

    kw = dict()
    if x.value in discrete:
        kw['x_range'] = sorted(set(xs))
    if y.value in discrete:
        kw['y_range'] = sorted(set(ys))
    kw['title'] = "%s vs %s counts" % (x_title, y_title)

    #if x.value in discrete:
    #    p.xaxis.major_label_orientation = pd.np.pi / 4

    sz = 9
    if size.value != 'None':
        #try:
        #    groups = pd.qcut(df[size.value].values, bins=[-0.2, 0.2, 0.4, 0.6, 0.8, 1.1])
        #    sz = [SIZES[xx] for xx in groups.codes]
        #except:
            groups = pd.cut(df[size.value].values, len(SIZES))
            sz = [SIZES[xx] for xx in groups.codes]

    c = "#31AADE"
    if color.value != 'None':
        if np.unique(df[color.value].values).shape[0]>4:
            COLORS = Greys4
            #groups = pd.qcut(df[color.value].values, len(COLORS))
            groups = pd.cut(df[color.value].values, len(COLORS))
            c = [COLORS[xx] for xx in groups.codes]
        elif np.unique(df[color.value].values).shape[0]>3:#4colors
            COLORS = [ "blue","green","red","darkorange"]
            #groups = pd.qcut(df[color.value].values, len(COLORS))
            groups = pd.cut(df[color.value].values, len(COLORS))
            c = [COLORS[xx] for xx in groups.codes]
        elif np.unique(df[color.value].values).shape[0]>2:#3colors
            COLORS = Greys3
            #groups = pd.qcut(df[color.value].values, len(COLORS))
            groups = pd.cut(df[color.value].values, len(COLORS))
            c = [COLORS[xx] for xx in groups.codes]
        elif np.unique(df[color.value].values).shape[0]>1:#2colors
            COLORS = ["red", "blue"]
            #groups = pd.qcut(df[color.value].values, len(COLORS))
            groups = pd.cut(df[color.value].values, len(COLORS))
            c = [COLORS[xx] for xx in groups.codes]


    source = ColumnDataSource(
        data = dict(
            x=xs,
            y=ys,
            #y2= ys2,
            module  = df.MouseHoverlabel,
            #weights = df.MouseHoverlabel2,
            #imgs = ['http://127.0.0.1/a.png']
            )
        )


    hover = HoverTool(
        tooltips=[
        ("module","@module"),
        ("%s, %s" % (x_title, y_title),"@x, @y"),
        #("weight","@weights"),
        #("EWtest", "<img src=@imgs />")
        ]
        )

    if y2.value != 'None':
        hover = HoverTool(
            tooltips=[
            ("module","@module"),
            ("%s, %s, %s" % (x_title, y_title, y2_title),"@x, @y, @y2"),
            ]
            )


    #p = figure(plot_height=600, plot_width=800, tools=['pan,box_zoom,reset,lasso_select',hover], **kw)
    p = figure(plot_height=600, plot_width=600, tools=['pan,box_zoom,reset,wheel_zoom,xwheel_zoom,ywheel_zoom,save,lasso_select',hover], **kw)
    p.xaxis.axis_label = x_title
    p.yaxis.axis_label = y_title

    #p.select(LassoSelectTool).select_every_mousemove = False

    #p.circle(x=xs, y=ys, color=c, size=sz, line_color="white", alpha=0.6, hover_color='white', hover_alpha=0.5,source = source)
    r = p.circle('x', 'y', color=c, size=sz, line_color="white", alpha=0.6, hover_color='white', hover_alpha=0.5,source = source)



    if y2.value != 'None':
        ys2 = df[y2.value].values
        source = ColumnDataSource(
            data = dict(
                x=xs,
                y=ys,
                y2= ys2,
                module  = df.MouseHoverlabel,
                #weights = df.MouseHoverlabel2,
                #imgs = ['http://127.0.0.1/a.png']
                )
            )
        p.triangle('x', 'y2', size=sz, line_color="white", alpha=0.6, hover_color='white', hover_alpha=0.5,source = source)


    r.glyph.line_color = 'black'
    #return p
    return p


def update(attr, old, new):
    layout.children[1] = create_figure()


#x = Select(title='X-Axis', value='mpg', options=columns)
x = Select(title='X-Axis', value='%s' % columns[1], options=columns)
#x = Select(title='X-Axis', value='None', options=['None'] + columns)
x.on_change('value', update)

#y = Select(title='Y-Axis', value='hp', options=columns)
y = Select(title='Y-Axis', value='%s' % columns[2], options=columns)
#y = Select(title='Y-Axis', value='None', options=['None'] + columns)
y.on_change('value', update)



y2 = Select(title='Y2-Axis', value='None', options=['None'] + columns)
#y2 = Select(title='Y2-Axis', value='%s' % columns[3], options=['None'] + columns)
y2.on_change('value', update)

#size = Select(title='Size', value='None', options=['None'] + quantileable)
#size = Select(title='Size', value='%s' % quantileable[3], options=['None'] + quantileable)
#size = Select(title='Size', value='%s' % columns[3], options=['None'] + quantileable)
size = Select(title='Size', value='None', options=['None'] + columns)
size.on_change('value', update)

#color = Select(title='Color', value='None', options=['None'] + quantileable)
color = Select(title='Color', value='None', options=['None'] + columns)
#color = Select(title='Color', value='%s' % columns[4], options=['None'] + columns)
color.on_change('value', update)


#controls = widgetbox([x, y, size], width=200)
controls = widgetbox([x, y, y2, size, color], width=150)
#controls = widgetbox([x, y, size, color])
layout = row(controls, create_figure())



curdoc().add_root(layout)
curdoc().title = "Data Cross Filter"