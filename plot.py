import sys
import glob
import time
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.mlab as mlab
import matplotlib.cbook as cbook
import matplotlib.gridspec as gridspec
import scipy.spatial as spatial
from matplotlib.widgets import Button
import ephem as e

totalplots = 4*6
G = gridspec.GridSpec(1+totalplots/4, 4)

def fmt1(x, y):
    pngfn = sorted(glob.glob('./201*/'+ moduleDic[round(x)][0:3]+'*.png'),reverse=True)
    print moduleDic[round(x)][0:3], 'ploting'
    for i in reversed(range(totalplots)):
        plt.subplot(G[i/4, i%4])#.cla()
        im[i].set_data(Image.open(pngfn[i]))
        yaxislael = pngfn[i].split('/')[2][0:5] +'\n'+ pngfn[i][2:2+10]+'\n'+ pngfn[i][2+11:21]
        if pngfn[i].split('/')[2][0:5] == module[int(round(x))][:5]:
            plt.ylabel(yaxislael,fontsize = 14, color = 'blue')
        else:
            #plt.ylabel(pngfn[i].split('/')[1][5:10]+' '+pngfn[i].split('/')[2][0:5],fontsize = 14, color = 'black')
            plt.ylabel(yaxislael,fontsize = 14, color = 'black')

    print 'ok!'
    fig.canvas.draw()
    return module[int(round(x))][:5]


def fmt2(x, y):
    return module[int(round(x))][:5]

class DataCursor(object):
    """A simple data cursor widget that displays the x,y location of a
    matplotlib artist when it is selected."""
    #def __init__(self, artists, tolerance=5, offsets=(-20, 20), template='x: %0.2f\ny: %0.2f', display_all=False):
    def __init__(self, artists, tolerance=20, offsets=(-20, 20), template= fmt1, display_all=False):
        """Create the data cursor and connect it to the relevant figure.
        "artists" is the matplotlib artist or sequence of artists that will be 
            selected. 
        "tolerance" is the radius (in points) that the mouse click must be
            within to select the artist.
        "offsets" is a tuple of (x,y) offsets in points from the selected
            point to the displayed annotation box
        "template" is the format string to be used. Note: For compatibility
            with older versions of python, this uses the old-style (%) 
            formatting specification.
        "display_all" controls whether more than one annotation box will
            be shown if there are multiple axes.  Only one will be shown
            per-axis, regardless. 
        """
        self.template = template
        self.offsets = offsets
        self.display_all = display_all
        if not cbook.iterable(artists):
            artists = [artists]
        self.artists = artists
        self.axes = tuple(set(art.axes for art in self.artists))
        self.figures = tuple(set(ax.figure for ax in self.axes))

        self.annotations = {}
        for ax in self.axes:
            self.annotations[ax] = self.annotate(ax)

        for artist in self.artists:
            artist.set_picker(tolerance)
        for fig in self.figures:
            fig.canvas.mpl_connect('pick_event', self)

    def annotate(self, ax):
        """Draws and hides the annotation box for the given axis "ax"."""
        annotation = ax.annotate(self.template, xy=(0, 0), ha='right',
                xytext=self.offsets, textcoords='offset points', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                )
        annotation.set_visible(False)
        return annotation

    def __call__(self, event):
        """Intended to be called through "mpl_connect"."""
        # Rather than trying to interpolate, just display the clicked coords
        # This will only be called if it's within "tolerance", anyway.
        x, y = event.mouseevent.xdata, event.mouseevent.ydata
        annotation = self.annotations[event.artist.axes]
        if x is not None:
            if not self.display_all:
                # Hide any other annotation boxes...
                for ann in self.annotations.values():
                    ann.set_visible(False)
            # Update the annotation in the current axis..
            annotation.xy = x, y
            #annotation.set_text(self.template % (x, y))
            annotation.set_text(self.template(x, y))
            annotation.set_visible(True)
            #event.canvas.draw()

class FollowDotCursor(object):
    """Display the x,y location of the nearest data point.
    http://stackoverflow.com/a/4674445/190597 (Joe Kington)
    http://stackoverflow.com/a/13306887/190597 (unutbu)
    http://stackoverflow.com/a/15454427/190597 (unutbu)
    """
    def __init__(self, ax, x, y, tolerance=5, formatter=fmt2, offsets=(-20, -20)):
        try:
            x = np.asarray(x, dtype='float')
        except (TypeError, ValueError):
            x = np.asarray(mdates.date2num(x), dtype='float')
        y = np.asarray(y, dtype='float')
        #print x,y
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        self._points = np.column_stack((x, y))
        self.offsets = offsets
        y = y[np.abs(y-y.mean()) <= 3*y.std()]
        self.scale = x.ptp()
        #self.scale = y.ptp() / self.scale if self.scale else 1
        self.scale = y.ptp() *2
        self.tree = spatial.cKDTree(self.scaled(self._points))
        self.formatter = formatter
        self.tolerance = tolerance
        self.ax = ax
        self.fig = ax.figure
        self.ax.xaxis.set_label_position('top')
        self.dot = ax.scatter(
            [x.min()], [y.min()], s=130, color='green', alpha=0.7)
        self.annotation = self.setup_annotation()
        plt.connect('motion_notify_event', self)
        #self.fig.canvas.mpl_connect('key_press_event', self.press)
    '''
    def press(self, event):
        print('press', event.key)
        sys.stdout.flush()
        if event.key == 'q':
            visible = xl.get_visible()
            xl.set_visible(not visible)
            fig.canvas.draw()
    '''
    def scaled(self, points):
        points = np.asarray(points)
        return points * (self.scale, 1)

    def __call__(self, event):
        ax = self.ax
        # event.inaxes is always the current axis. If you use twinx, ax could be
        # a different axis.
        if event.inaxes == ax:
            x, y = event.xdata, event.ydata
        elif event.inaxes is None:
            return
        else:
            inv = ax.transData.inverted()
            x, y = inv.transform([(event.x, event.y)]).ravel()
        annotation = self.annotation
        x, y = self.snap(x, y)
        annotation.xy = x, y
        annotation.set_text(self.formatter(x, y))
        self.dot.set_offsets((x, y))
        bbox = ax.viewLim
        event.canvas.draw()

    def setup_annotation(self):
        """Draw and hide the annotation box."""
        annotation = self.ax.annotate(
            '', xy=(0, 0), ha = 'right',
            xytext = self.offsets, textcoords = 'offset points', va = 'bottom',
            bbox = dict(
                boxstyle='round,pad=0.5', fc='yellow', alpha=0.75),
            arrowprops = dict(
                arrowstyle='->', connectionstyle='arc3,rad=0'))
        return annotation

    def snap(self, x, y):
        """Return the value in self.tree closest to x, y."""
        dist, idx = self.tree.query(self.scaled((x, y)), k=1, p=1)
        try:
            return self._points[idx]
        except IndexError:
            # IndexError: index out of bounds
           return self._points[0]


def getlst(utc):
    import subprocess
    cmd = "mopsr_getlst "+utc
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    null, lst = proc.communicate()
    lst = lst[6:19]
    #lst = hms2hrs(lst)
    return lst

def J2000_to_radec(ra,dec,utc):
    eq = e.Equatorial(ra, dec, epoch=e.J2000)
    eq2 = e.Equatorial(eq,epoch = utcaj_2_utceb(utc) )
    return eq2.ra, eq2.dec   


def utcaj_2_utceb(utc):
    utc = utc.split("-")
    utc = utc[0]+"/"+utc[1]+"/"+utc[2]+" "+utc[3]
    return utc





fn = sorted(glob.glob('*.txt'))

print
for i in range(len(fn)):
    if i >= len(fn)-3:
        print i-(len(fn)-4),fn[i]
foo = len(fn)-1
print 'The chosen one is ', fn[foo]
print 

utc = fn[foo][0:19]
lst =  getlst(utc)
#lst = "08:03:37.697"
#print lst
#sys.exit()
#assuming the calibrator is J0835-4510
#start_time_ahead_min = (8 - int(lst[0:2]))*60.+ 35 - int(lst[3:5])


ra, dec = J2000_to_radec('08:35:20.61149', '-45:10:34.8751', utc)

start_time_ahead_min = (int(str(ra).split(':')[0])-int(str(lst).split(':')[0]))*60.+ (int(str(ra).split(':')[1])-int(str(lst).split(':')[1])) + (float(str(ra).split(':')[2])-float(str(lst).split(':')[2]))/60  + 0.790556 #HA is not zero in stationary mode


integration_time = 20 #seconds
phase_width_chk = 0.75
#print fn[0]
data = []
txt = open(fn[foo])
for line in txt.readlines():
        line = line.replace('[','')
        line = line.replace(']','')
        if (line[0][0]) == '0':
                data.append(line)
        if (line[0][0]).isdigit():
                if int(line[0][0])>0:
                        data.append(line)

order = []
module = []
maxsnr = []
height = []
center_time = []
center_phase = []
time_width = []
phase_width = []

#for i in range(16):
#    print i, data[0].split(',')[i]
#print (data[0].split(',')[13])
#sys.exit()

#print data[0].split(',')
#for i in range(len(data[0].split(','))):
#    print i,data[0].split(',')[i]
#sys.exit()


for i in range(len(data)):
        order.append(       float(data[i].split(',')[0]  ) )
        module.append(            data[i].split(',')[1][2:-4] )
        maxsnr.append(            data[i].split(',')[18] )
        height.append(      float(data[i].split(',')[12] ) )
        center_time.append( float(data[i].split(',')[13] ) )
        center_phase.append(float(data[i].split(',')[14] ) )
        time_width.append(  float(data[i].split(',')[15] ) )
        phase_width.append( float(data[i].split(',')[16] ) )
        if i == int(0.5*len(data)-1):
            module = module[::-1]
            maxsnr = maxsnr[::-1]
            height = height[::-1]
            center_time = center_time[::-1]
            center_phase = center_phase[::-1]
            time_width = time_width[::-1]
            phase_width = phase_width[::-1]

#print i, module
#sys.exit()


# W to E
module = module[::-1]
maxsnr = maxsnr[::-1]
height = height[::-1]
center_time = center_time[::-1]
center_phase = center_phase[::-1]
time_width = time_width[::-1]
phase_width = phase_width[::-1]


#print height
#print float(maxsnr)

height = map(float,maxsnr)


data = np.array([order, height, center_time, center_phase, time_width, phase_width])


moduleDic ={}
for i in range(len(module)):
    for jj in range(2):
        moduleDic[((data[jj]).tolist())[i]] = module[i]

senDic ={} #sensitivity dictionary
for i in range(len(module)):
        senDic[module[i]]  = ((data[1]).tolist())[i]

tcDic ={} #center time dictionary
for i in range(len(module)):
        tcDic[module[i]]  = ((data[2]).tolist())[i]

moduleDicgood = {}
for i in data[0][data[5]<phase_width_chk]:
    moduleDicgood[i] = moduleDic[i]


print 'bad/good ratio = ', 352-len(data[1][data[5]<phase_width_chk]),'/',len(data[1][data[5]<phase_width_chk])
print

fig = plt.figure(figsize=(15, 10))
#G = gridspec.GridSpec(4+3, 4)
#print data.shape
#print data[:,0]
#sys.exit()


#print data[2][data[5]<phase_width_chk]
data[2][data[2]>180] = 180
data[2][data[2]<0] = 0
data[2][data[5]>phase_width_chk]= start_time_ahead_min*60./integration_time #90
data[4][data[5]>phase_width_chk]= 0
data[4][data[4]>90]= 90
#print data[2][data[5]<phase_width_chk]


radiusfactor = 1.
data[0] /=radiusfactor

ax = plt.subplot(G[-1,:])
plt.subplot(G[-1,:])
plt.ylabel('Elapsed Time (min)')
x = (data[0][data[5]<=phase_width_chk]).tolist()
y = (1./60*integration_time*data[2][data[5]<=phase_width_chk]).tolist()
plt.errorbar(x, y, yerr= 0.5/3*data[4][data[5]<phase_width_chk],fmt = 'k.')


x = (data[0]).tolist()
y = (1./60*integration_time*data[2]).tolist()
scat = plt.scatter(x, y, s=0.01)

DataCursor(scat)


plt.axvline(175.5/radiusfactor, color='b', linestyle='dashed', linewidth=2)
plt.xlim(-1.3,352.7/radiusfactor)
plt.ylim(0, 2*start_time_ahead_min)

#plt.subplots_adjust(top = 0.97, hspace = 0.3)
plt.subplots_adjust(top = 1, hspace = 0.001, bottom = 0.05)
#plt.subplots_adjust(top = 0.97, hspace = 0.03,bottom = 0.1)



x = (data[0]).tolist()
y = (1./3*data[2]).tolist()
cursor = FollowDotCursor(ax,x, y, tolerance=20)

x = []
for i in range(44,0,-1): x.append('W%02i' % i)
for i in range(1,45): x.append('E%02i' % i)

#plt.subplots_adjust(wspace=0.5, hspace=0.5)
xp = (np.arange(89)*4+1.5)/radiusfactor
plt.xticks(xp,x,rotation='vertical')
#plt.xticks([])


modulebutton = 'E01'
pngfn = sorted(glob.glob('./201*/'+ modulebutton+'*.png'),reverse=True)

im=[]
for i in range(totalplots):
    plt.subplot(G[i/4, i%4])#.cla()
    im.append( plt.imshow(Image.open(pngfn[i])))
    plt.xticks([])
    plt.yticks([])
    plt.ylabel(pngfn[i].split('/')[1][5:10]+' '+pngfn[i].split('/')[2][0:5],fontsize = 14)


#plt.tight_layout()
plt.show()
