import sys
import numpy as np
import psrchive
import matplotlib.pyplot as plt
import glob
from scipy import optimize
import os
from scipy.optimize import leastsq
from scipy.signal import argrelextrema
import ephem as e

def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p

def Decimate_2D(ts, ndown=2):
    if ndown==1:
       return ts
    ts = ts.T
    ncols = len(ts)
    n_rep = ncols / ndown
    ts_ds = np.array([ts[i::ndown][0:n_rep] for i in range(ndown)]).mean(0)
    return ts_ds.T

def snr(x):
    y = np.roll(x, -np.where((x == x.max().mean()))[0][0])
    y= y[velaPhasewidth:-velaPhasewidth]
    #print np.where((x == x.max().mean()))[0][0]
    #plt.clf()
    #plt.plot(y)
    #plt.plot(x)
    #plt.show()
    return (x - np.median(y))/y.std()

def jstd(x):
    y = np.roll(x, -np.where((x == x.max().mean()))[0][0])
    y= y[velaPhasewidth:-velaPhasewidth]
    #print np.where((x == x.max().mean()))[0][0]
    #plt.clf()
    #plt.plot(y)
    #plt.plot(x)
    #plt.show()
    return y.std()



def gauss(x, center,FWHM,high,base):
    return high*exp(-(x-center)**2/(2*(FWHM/2.355)**2))+base


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



nargs = len(sys.argv)-1
if (nargs>0):
    txtfoo = sys.argv[1]
    openfoo= sys.argv[1]


'''
#This part used to dig out the folder containning Module Mode observation.
# lookin present archives
chkobsinfo = sorted(glob.glob('/data/mopsr/archives/*/J0835-4510'))

# or lookin old archives
#chkobsinfo = sorted(glob.glob('/data/mopsr/old_archives/*/J0835-4510'))

for i in range(len(chkobsinfo)):
    #if chkobsinfo[i] == '/data/mopsr/archives/2016-11-10-18:49:11/J0835-4510':
    lines = [line.rstrip('\n') for line in open(chkobsinfo[i][:-10]+'obs.info')]
    for j in lines:
        if j[:2] == 'MB':
            if j[-4:] == 'true':
                fn = sorted(glob.glob(chkobsinfo[i][:-10]+'FB/'+'*-*_*_BEAM_*/*-*_*-*_BEAM_*.ar'))
                if len(fn) !=0:
                    print chkobsinfo[i][:-10]#, fn[0]
'''
#sys.exit()

if nargs > 0:
    #if txtfoo =='a':# <20 mins
    #    fn = sorted(glob.glob('/data/mopsr/archives/2016-12-14-17:14:11/FB/*BEAM*/*BEAM*.ar') )
    #if txtfoo =='1':#17 mins
    #    fn = sorted(glob.glob('/data/mopsr/old_archives/2016-10-17-20:46:24/FB/*-*_*_BEAM_*/*-*_*-*_BEAM_*.ar') )
    #"""
    if txtfoo =='2':#1 hour
        fn = sorted(glob.glob('/data/mopsr/old_archives/2016-10-19-20:13:12/FB/*-*_*_BEAM_*/*-*_*-*_BEAM_*.ar') )
    if txtfoo =='3':#1 hour
        fn = sorted(glob.glob('/data/mopsr/old_archives/2016-11-10-18:49:11/FB/*-*_*_BEAM_*/*-*_*-*_BEAM_*.ar') )
    if txtfoo =='4':#1 hour
        fn = sorted(glob.glob('/data/mopsr/old_archives/2016-11-17-18:17:26/FB/*-*_*_BEAM_*/*-*_*-*_BEAM_*.ar') )
    if txtfoo =='5':#1 hour
        fn = sorted(glob.glob('/data/mopsr/old_archives/2016-11-21-18:03:55/FB/*-*_*_BEAM_*/*-*_*-*_BEAM_*.ar') )
    if txtfoo =='6':#1 hour
        fn = sorted(glob.glob('/data/mopsr/old_archives/2016-11-22-17:58:29/FB/*-*_*_BEAM_*/*-*_*-*_BEAM_*.ar') )
    if txtfoo =='7':#1 houry
        fn = sorted(glob.glob('/data/mopsr/old_archives/2016-12-01-17:22:41/FB/*BEAM*/*BEAM*.ar') )
    if txtfoo =='8':#1 houry
        fn = sorted(glob.glob('/data/mopsr/old_archives/2016-12-06-17:02:17/FB/*BEAM*/*BEAM*.ar') )
    if txtfoo =='9':#1 houry
        fn = sorted(glob.glob('/data/mopsr/old_archives/2017-01-02-15:14:59/FB/*BEAM*/*BEAM*.ar') )
    #if txtfoo =='10':#0.8 houry
    #    fn = sorted(glob.glob('/data/mopsr/archives/2017-01-09-14:57:51/FB/*BEAM*/*BEAM*.ar') )
    if txtfoo =='11':#1 houry
        fn = sorted(glob.glob('/data/mopsr/old_archives/2017-01-15-14:24:39/FB/*BEAM*/*BEAM*.ar') )
    #if txtfoo =='12':#0.5 houry
    #    fn = sorted(glob.glob('/data/mopsr/archives/2017-01-16-15:40:58/FB/*BEAM*/*BEAM*.ar') )
    if txtfoo =='13':#1 houry
        fn = sorted(glob.glob('/data/mopsr/old_archives/2017-01-27-13:38:20/FB/*BEAM*/*BEAM*.ar') )
    if txtfoo =='14':#1 houry
        fn = sorted(glob.glob('/data/mopsr/old_archives/2017-02-08-12:51:24/FB/*BEAM*/*BEAM*.ar') )
    if txtfoo =='15':#1 houry
        fn = sorted(glob.glob('/data/mopsr/old_archives/2017-02-15-12:24:05/FB/*BEAM*/*BEAM*.ar') )
    if txtfoo =='16':#1 houry
        fn = sorted(glob.glob('/data/mopsr/archives/2017-03-01-11:28:59/FB/*BEAM*/*BEAM*.ar') )
    if txtfoo =='17':#1 houry
        fn = sorted(glob.glob('/data/mopsr/archives/2017-03-06-11:11:57/FB/*BEAM*/*BEAM*.ar') )
    if txtfoo =='18':#1 houry
        fn = sorted(glob.glob('/data/mopsr/archives/2017-03-07-10:55:57/FB/*BEAM*/*BEAM*.ar') )
    if txtfoo =='19':#1 houry
        fn = sorted(glob.glob('/data/mopsr/archives/2017-03-08-11:03:05/FB/*BEAM*/*BEAM*.ar') )
    #"""
    else:fn = sorted(glob.glob('/data/mopsr/archives/%s/FB/*BEAM*/*BEAM*.ar' % sys.argv[1]) )

if len(fn) == 0:
    print ''
    print ''
    print sys.argv[1], 'is an invalid UTC start time, not found under /data/mopsr/archives/'
    print ''
    print ''
    sys.exit()

txtfoo = open(fn[0].split('/')[4]+'.txt','w+')

if not os.path.isdir(fn[0].split('/')[4]):
    os.makedirs(fn[0].split('/')[4])


factor = 4.
velaPhasewidth = 15
integrationtime = 20 #sec
channels = 40.

utc = fn[0].split('/')[4]
lst =  getlst(utc)

ra, dec = J2000_to_radec('08:35:20.61149', '-45:10:34.8751', utc)
#print int(str(lst).split(':')[0]), int(str(lst).split(':')[1]), float(str(lst).split(':')[2])
#print int(str(ra).split(':')[0]), int(str(ra).split(':')[1]), float(str(ra).split(':')[2])
#assuming the calibrator is J0835-4510
#start_time_ahead_min = (8 - int(lst[0:2]))*60.+ 35 - int(lst[3:5])- 0.790556 #HA is not zero in stationary mode
start_time_ahead_min = (int(str(ra).split(':')[0])-int(str(lst).split(':')[0]))*60.+ (int(str(ra).split(':')[1])-int(str(lst).split(':')[1])) + (float(str(ra).split(':')[2])-float(str(lst).split(':')[2]))/60 + 0.790556 #HA is not zero in stationary mode

for i in range(177+4*9+2,177+4*9+3):
	j = fn[i][::-1].index('/')
	foo = fn[i][-1*j:-3]
	#print foo,fn[i]
        #sys.exit()
	arch = psrchive.Archive_load(fn[i])
	arch.dedisperse()
	arch.remove_baseline()
	data = arch.get_data()
	data = (data[:,0,:,:].mean(1)).mean(0)
	#print data.shape
	#print data
	roll = data.shape[0]/2 - np.where(data == data.max())[0][0]
        data = np.roll(data,roll)
	#print np.where(data == data.max())
	#sys.exit()


for i in range(len(fn)):
#for i in range(29*4-4,29*4-3): #give it a test
#for i in range(177+4*9+2,177+4*9+3):
#for i in range(0,1):
	j = fn[i][::-1].index('/')
	foo = fn[i][-1*j:-3]
	#print foo,fn[i]
        #sys.exit()
	arch = psrchive.Archive_load(fn[i])
	arch.dedisperse()
	arch.remove_baseline()
	data = arch.get_data()
        data = np.roll(data,roll)

        #sys.exit()
	plt.subplot(2,2,3)
	plt.imshow(data[:,0,:,:].mean(1), origin = 'low', aspect = 'auto',extent = [0,1,0,data.shape[0]*1.*integrationtime/60])
        snrtransit = snr(data[start_time_ahead_min*60./integrationtime,0,:,:].mean(0))
        y=1*snrtransit
        x=np.arange(len(y))
        v0=[1, np.where(y == y.max().mean())[0][0], velaPhasewidth, 0.1]
        gauss_fit = lambda p, x: p[0]*np.exp(-(x-p[1])**2/(2*(p[2]/2.355)**2))+p[3]
        e_gauss_fit = lambda p, x, y: (gauss_fit(p,x) -y)
        out = leastsq(e_gauss_fit, v0[:], args=(x, y))
        ccc = gauss_fit(out[0],x)
        #snrtransit = ccc.max()
        #plt.figure()
        #plt.plot(snrtransit)
        #plt.axvline(x = out[0][1]-velaPhasewidth)
        #plt.axvline(x = out[0][1]+velaPhasewidth)
        #plt.show()
        snrtransit = snrtransit[out[0][1]-velaPhasewidth:out[0][1]+velaPhasewidth].mean()*np.sqrt(2*velaPhasewidth)
        
        stdtransit = jstd(data[start_time_ahead_min*60./integrationtime,0,:,:].mean(0))
	plt.xlabel('Pulse phase (Period)')
	plt.ylabel('Elapsed Time (min)')
        plt.axhline(start_time_ahead_min, color='b', linestyle='dashed', linewidth=2)


	#plt.show()

	data = data[:,0,:,:].mean(1)

        if data.mean() ==0:
	    print i,', mean is zero,', foo, ', NA, NA , NA, NA, NA'
            plt.clf()
            plt.imshow(data,origin = 'low')
            plt.xticks([],[])
            plt.yticks([],[])
            #plt.show()
            plt.savefig('./'+fn[0].split('/')[4]+'/'+foo)
            plt.clf()
            continue

        dataDec = (Decimate_2D(data,10))**factor

        params = fitgaussian(dataDec)

        if np.isnan(params).any():
            print i, ',2D gaussian fail,', foo, ', NA, NA , NA, NA, NA'
        #    #continue

        fit = gaussian(*params)
        plt.subplot(2,2,2)
        if fit(*np.indices(dataDec.shape)).mean() != 0:
            plt.contour(fit(*np.indices(dataDec.shape)), cmap=plt.cm.copper)
        plt.axhline(start_time_ahead_min*60./integrationtime, color='b', linestyle='dashed', linewidth=2)
        #plt.xlim(0,1)
        #plt.ylim(0,60)
        plt.xticks([],[])
        plt.yticks([],[])

        y = data.mean(0)
        y = snr(y)
        snrarea = y.sum()
        x = np.arange(y.shape[0])
        #if params[2]*10>x.shape[0]:
        #     v0=[1, np.where(y == y.max().mean())[0][0], velaPhasewidth, 0.1]
        #elif params[2] <0:
        #     v0=[1, np.where(y == y.max().mean())[0][0], velaPhasewidth, 0.1]
        #else:v0=[1, params[2]*10, velaPhasewidth, 0.1]
        v0=[1, np.where(y == y.max().mean())[0][0], velaPhasewidth, 0.1]
        gauss_fit = lambda p, x: p[0]*np.exp(-(x-p[1])**2/(2*(p[2]/2.355)**2))+p[3]
        e_gauss_fit = lambda p, x, y: (gauss_fit(p,x) -y)
        #out = leastsq(e_gauss_fit, v0[:], args=(x, y), maxfev=1e5, full_output=1)
        out = leastsq(e_gauss_fit, v0[:], args=(x, y))
        ccc = gauss_fit(out[0],x)
        maxsnr=ccc.max()

        plt.subplot(2,2,1)
	#plt.xlabel('S/N')
	plt.ylabel('S/N')
        plt.plot(x, ccc,'r',label='Fitted')
        plt.plot(x, y, label='Real Data')
        plt.plot(x, snr( arch.get_data()[start_time_ahead_min*60/integrationtime,0,:,:].mean(0)) )
        #plt.legend()
        plt.xticks([],[])
        #plt.yticks([],[])
        plt.axvline(out[0][1]-15, color='b', linestyle='dashed', linewidth=2)
        plt.axvline(out[0][1]+15, color='b', linestyle='dashed', linewidth=2)
        plt.xlim(0,x.max())
        plt.ylim(-5,35.)


        #print out[0][1]
        if out[0][1]> y.shape[0]:out[0][1] = np.where(data.mean(1) == data.mean(1).max())[0][0]
        if out[0][1]< 0:         out[0][1] = np.where(data.mean(1) == data.mean(1).max())[0][0]
        #print out[0][1]

        #std = []
        #print data.shape
        #for stdi in range(data.shape[1]):
        #    std.append( data[:,stdi].std() )
        #print len(std)
        #std = ( np.asarray(std) ).mean()
        #print std
        #y = data[:, out[0][1]-velaPhasewidth:out[0][1]+velaPhasewidth].mean(1)
        y = np.zeros((data.shape[0]))
        for j in range(data.shape[0]):
            y[j] = snr(data[j,:])[out[0][1]-velaPhasewidth:out[0][1]+velaPhasewidth].mean()*np.sqrt(2*velaPhasewidth)
        #y = y/std/np.sqrt(2*velaPhasewidth)
        x = np.arange(y.shape[0])
        #for i in range(len(x)):
        #    print y[i], x[i]
        #plt.clf()
        #plt.plot(y,x)
        #plt.show()
        #sys.exit()


        v1=[0.2, 0.2, 60, 120, 30, 30, 0.1]
        gauss_fit2 = lambda p, x: p[0]*np.exp(-(x-p[2])**2/(2*(p[4]/2.355)**2))+p[1]*np.exp(-(x-p[3])**2/(2*(p[5]/2.355)**2))+p[6]
        e_gauss_fit2 = lambda p, x, y: (gauss_fit2(p,x) -y)
        gout2 = leastsq(e_gauss_fit2, v1[:], args=(x, y), maxfev=1e5, full_output=1)
        gccc2 = gauss_fit2(gout2[0],x)

        v1=[1.2, 2.2, 3., 4.,.1]
        poly_fit = lambda p, x: p[0]*x**4 + p[1]*x**3 + p[2]*x**2 + p[3]*x + p[4]
        e_poly_fit = lambda p, x, y: (poly_fit(p,x) -y)
        out2 = leastsq(e_poly_fit, v1[:], args=(x, y), maxfev=1e5, full_output=1)
        ccc2 = poly_fit(out2[0],x)

        plt.subplot(2,2,4)
	plt.xlabel('S/N (20sec/module)')
	#plt.ylabel('Elapsed Time (min)')
        #plt.plot(y, x, label='Real Data')
        plt.scatter(y, x, label='Real Data')
        plt.plot(gccc2, x, label='Fitted Data')
        #plt.axhline(out2[0][2], color='b', linestyle='dashed', linewidth=2)
        plt.axhline(start_time_ahead_min*60./integrationtime, color='b', linestyle='dashed', linewidth=2)
        plt.xlim(-5, 20)
        plt.ylim(0,x.max())
        #plt.xlim(y.min(),y.max())
        #plt.xticks([],[])
        plt.yticks([],[])
        #plt.legend()

        #print i, ',', foo, maxsnr,',', out[0].tolist(), ',', out2[0].tolist(), ',', "%.4f, %.2f, %.2f, %.2f, %.2f"% (abs(params)[0]**1./factor, params[1], params[2], abs(params[3]), abs(params[4]) ), maxsnr*ccc2[start_time_ahead_min*60./integrationtime]/ccc2.max(), len(argrelextrema(ccc2,np.greater)[0]), len(argrelextrema(ccc2,np.less)[0]) #the intensity at transit, the len of local max/min

        #print i, ',', foo, snrarea,',', out[0].tolist(), ',', out2[0].tolist(), ',', "%.4f, %.2f, %.2f, %.2f, %.2f"% (abs(params)[0]**1./factor, params[1], params[2], abs(params[3]), abs(params[4]) ), snrarea*gccc2[start_time_ahead_min*60./integrationtime]/gccc2.max(), len(argrelextrema(ccc2,np.greater)[0]), len(argrelextrema(ccc2,np.less)[0]) #the intensity at transit, the len of local max/min

        #print i, ',', foo, snrarea,',', out[0].tolist(), ',', out2[0].tolist(), ',', "%.4f, %.2f, %.2f, %.2f, %.2f"% (abs(params)[0]**1./factor, params[1], params[2], abs(params[3]), abs(params[4]) ), gccc2[start_time_ahead_min*60./integrationtime]*np.sqrt(channels), len(argrelextrema(ccc2,np.greater)[0]), len(argrelextrema(ccc2,np.less)[0]) #the intensity at transit, the len of local max/min

        #print i, ',', foo, snrarea,',', out[0].tolist(), ',', out2[0].tolist(), ',', "%.4f, %.2f, %.2f, %.2f, %.2f"% (abs(params)[0]**1./factor, params[1], params[2], abs(params[3]), abs(params[4]) ),gccc2.max()*np.sqrt(channels),  snrtransit, len(argrelextrema(ccc2,np.greater)[0]), len(argrelextrema(ccc2,np.less)[0]) ,stdtransit, (ccc.max()-out[0][-1])/gccc2.max()*gccc2[start_time_ahead_min*60./integrationtime],  ccc2[start_time_ahead_min*60./integrationtime]/((ccc2.max()))

        #print i, ',', foo, snrarea,',', out[0].tolist(), ',', out2[0].tolist(), ',', "%.4f, %.2f, %.2f, %.2f, %.2f"% (abs(params)[0]**1./factor, params[1], params[2], abs(params[3]), abs(params[4]) ),gccc2.max()*np.sqrt(channels),  snrtransit, len(argrelextrema(ccc2,np.greater)[0]), len(argrelextrema(ccc2,np.less)[0]) ,stdtransit,  ccc2[start_time_ahead_min*60./integrationtime]/((ccc2.max()))

        #textosave =  i, foo[0:-1], snrarea, out[0].tolist(),  out2[0].tolist(), abs(params)[0]**1./factor, params[1], params[2], abs(params[3]), abs(params[4]) ,gccc2.max()*np.sqrt(channels),  snrtransit, len(argrelextrema(ccc2,np.greater)[0]), len(argrelextrema(ccc2,np.less)[0]), stdtransit, ccc2[start_time_ahead_min*60./integrationtime]/((ccc2.max()))

        #print i, ',', foo, snrarea,',', out[0].tolist(), ',', out2[0].tolist(), ',', "%.4f, %.2f, %.2f, %.2f, %.2f"% (abs(params)[0]**1./factor, params[1], params[2], abs(params[3]), abs(params[4]) ),gccc2.max()*np.sqrt(channels),  gccc2[start_time_ahead_min*60./integrationtime]*np.sqrt(channels), len(argrelextrema(ccc2,np.greater)[0]), len(argrelextrema(ccc2,np.less)[0]) ,stdtransit,  ccc2[start_time_ahead_min*60./integrationtime]/((ccc2.max()))

        #textosave =  i, foo[0:-1], snrarea, out[0].tolist(),  out2[0].tolist(), abs(params)[0]**1./factor, params[1], params[2], abs(params[3]), abs(params[4]) ,gccc2.max()*np.sqrt(channels),  gccc2[start_time_ahead_min*60./integrationtime]*np.sqrt(channels), len(argrelextrema(ccc2,np.greater)[0]), len(argrelextrema(ccc2,np.less)[0]), stdtransit, ccc2[start_time_ahead_min*60./integrationtime]/((ccc2.max()))


        #print i, ',', foo, maxsnr , ccc2[start_time_ahead_min*60./integrationtime]/((ccc2.max())), len(argrelextrema(ccc2,np.greater)[0]), len(argrelextrema(ccc2,np.less)[0]), abs(params[4]), snrtransit

        #textosave =  i, foo[0:-1], maxsnr, ccc2[start_time_ahead_min*60./integrationtime]/((ccc2.max())), len(argrelextrema(ccc2,np.greater)[0]), len(argrelextrema(ccc2,np.less)[0]), abs(params[4]), snrtransit


        print i, ',', foo, snrarea,',', out[0].tolist(), ',', out2[0].tolist(), ',', "%.4f, %.2f, %.2f, %.2f, %.2f"% (abs(params)[0]**1./factor, params[1], params[2], abs(params[3]), abs(params[4]) ),gccc2.max(),  gccc2[start_time_ahead_min*60./integrationtime], len(argrelextrema(ccc2,np.greater)[0]), len(argrelextrema(ccc2,np.less)[0]) ,stdtransit,  gccc2[start_time_ahead_min*60./integrationtime]/((gccc2.max())), snrtransit,start_time_ahead_min

        textosave = i, foo[0:-1], snrarea, out[0].tolist(),  out2[0].tolist(), abs(params)[0]**1./factor, params[1], params[2], abs(params[3]), abs(params[4]) ,gccc2.max(),  np.abs( gccc2[start_time_ahead_min*60./integrationtime] ), len(argrelextrema(ccc2,np.greater)[0]), len(argrelextrema(ccc2,np.less)[0]), stdtransit, gccc2[start_time_ahead_min*60./integrationtime]/((gccc2.max())), snrtransit,start_time_ahead_min



        txtfoo.write(str(textosave)[1:-1]+'\n')


        #print fn[i].split('/')[4]+' '+fn[i].split('/')[6]
        plt.suptitle(fn[i].split('/')[4]+' '+fn[i].split('/')[6])
        plt.tight_layout()
        #plt.show()
        plt.savefig('./'+fn[0].split('/')[4]+'/'+foo)
	plt.clf()
        #'''
txtfoo.close()
