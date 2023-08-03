#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 17:49:19 2018

@author: uic-cs
"""
"""
bar chart
"""
import numpy as np
import matplotlib.pyplot as plt


N = 5
menMeans = (20.11, 35.11, 30, 35, 27)
womenMeans = (25, 32, 34, 20.11, 25)
menStd = (2, 3, 4, 1, 2)
womenStd = (3, 5, 2, 3, 3)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, menMeans, width, yerr=menStd)
p2 = plt.bar(ind, womenMeans, width,
             bottom=menMeans, yerr=womenStd)

plt.ylabel('Scores')
plt.title('Scores by group and gender')
plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('Men', 'Women'))

plt.show()




"""
side by side bar chart and  y is log scale
"""
from matplotlib import pylab
import plotly.plotly as py

log = plt.figure()

data = ((3,1000), (10,3), (100,30), (500, 800), (50,1))

pylab.xlabel("FOO")
pylab.ylabel("FOO")
pylab.title("Testing")
pylab.gca().set_yscale('log',basey=2)
pylab.gca()

dim = len(data[0])
w = 0.75
dimw = w / dim

x = pylab.arange(len(data))
for i in range(len(data[0])) :
    y = [d[i] for d in data]
    b = pylab.bar(x + i * dimw, y, dimw, bottom=0.001)
pylab.gca().set_xticks(x + w / 2)
pylab.gca().set_ylim( (0.001,1000))

plot_url = py.plot_mpl(log, filename='mpl-log')

"""
matrix *Matrix in different format
ft16 
      compare with result from fl32
bf16
"""

"""bf16"""
import matplotlib.pyplot as plt
import numpy as np
meanbf16 = (1.0299807108499846e-16, 1.0005838511168755e-14, 1.1950963490239624e-12, 1.1272752906135753e-10, 1.028557381663998e-08, 1.2085340173270424e-06, 0.00011512724105740743,0.010912770116108821, 1.1323429169072086, 110.88942918959286, 716.3451857369)
stdbf16= (3.9992527297248016e-17, 3.917957738351406e-15,4.622152331918473e-13, 4.193081432925387e-11,3.917261822082698e-09,4.5263828681121974e-07,4.778423331389008e-05,0.004230410979726093,0.47029042944475985,38.612740847788665,152.46862911433735)
"""fl16"""
meanfl16=(1.2482272087106862e-14,1.2488740880965015e-12, 1.330892930117689e-10, 1.321772672096623e-08, 4.4168879033876184e-08, 5.491942728772643e-08, 4.282116058123102e-06, 0.00042757416483919265, 0.0441565229746197,4.270095670297127, 27.566035988040795)
stdfl16=(3.724784049597544e-15,4.0055547388534926e-13, 3.8470904932610736e-11,4.293010184716444e-09,1.5668417318514176e-08,2.404282979048769e-08,2.4162727227632406e-06,0.00023474215791158054, 0.022167361347480152, 2.371666876257218,10.46048050260168)
x=(0.5*1e-7, 0.5*1e-6, 0.5*1e-5, 0.5*1e-4,0.5*1e-3,0.5*1e-2,0.5*1e-1, 0.5, 5, 50, 90)
xlog2 = np.log2(x)


ax = plt.subplot(111)
ind = np.arange(len(meanbf16))
p1=ax.bar(ind+0.1, meanbf16,width=0.2, yerr = stdbf16, color='b',align='center')
p2=ax.bar(ind-0.1, meanfl16,width=0.2,yerr= stdfl16, color='g',align='center')

plt.title("Matrix Matrix Multiplication Comparison")
plt.xlabel("exponent of 2")
plt.ylabel("difference(in absolute error)")
plt.xticks(ind, ('-24', '-20', '-17', '-14', '-10','-7','-4','-1','2','5','7'))
#plt.gca().set_xscale('log',basex=2)
plt.gca().set_yscale('log',basey=2)
plt.legend((p1[0], p2[0]), ('bf16', 'fl16'))

plt.savefig("/home/uic-cs/Desktop/summerIntern/mmcom.png")

"""
matrix*vector muliplication
ft16 
      compare with result from fl32
bf16

"""
meanbf16=( 8.093039487037003e-17,  7.217682050384324e-15, 7.143691022367287e-13, 7.816281830309086e-11,6.759604490664249e-09, 7.875662805506815e-07, 7.262668153561006e-05, 0.007114052768675299, 0.8525020249077818, 73.73884084850634, 502.05165198408673) 
stdbf16=(3.652968092290535e-17,3.614372868823265e-15,3.4519076626223775e-13,4.130491070582237e-11,3.4481752447417304e-09,4.0926819638305507e-07, 3.7143507731550034e-05,0.0031100926195324363,0.429411829509854 ,36.57797789601707,143.3954454929257)
x=(0.5*1e-7, 0.5*1e-6, 0.5*1e-5, 0.5*1e-4,0.5*1e-3,0.5*1e-2,0.5*1e-1, 0.5, 5, 50, 90)

"""fl16"""
meanfl16=(9.481022896978787e-15, 8.474322002756848e-13, 8.303463269250442e-11,9.033294914066946e-09, 3.055877180638138e-08, 3.908601823118965e-08, 2.7435053605986097e-06, 0.00027663054827506964, 0.03217891768732293, 2.946244603494236, 20.32958325320149)
stdfl16=(3.661712309505673e-15,3.5111187878390497e-13,3.427380327507404e-11,4.059742035863046e-09,1.5961574130786927e-08,2.0395690528228318e-08,1.7870740877728402e-06,0.00016887451025452045,
 0.02385911734788104 ,2.0826609576212594,10.67458864857729)

import matplotlib.pyplot as plt
import numpy as np
#import plotly.plotly as py


ax = plt.subplot(111)
ind = np.arange(len(meanbf16))
p1=ax.bar(ind+0.1, meanbf16,width=0.2, yerr = stdbf16, color='r',align='center')
p2=ax.bar(ind-0.1, meanfl16,width=0.2,yerr= stdfl16, color='g',align='center')

plt.title("Matrix Vector Multiplication Comparison")
plt.xlabel("exponent of 2")
plt.ylabel("difference(in absolute error")
plt.xticks(ind, ('-24', '-20', '-17', '-14', '-10','-7','-4','-1','2','5','7'))
#plt.gca().set_xscale('log',basex=2)
plt.gca().set_yscale('log',basey=2)
plt.legend((p1[0], p2[0]), ('bf16', 'fl16'))

plt.savefig("/home/uic-cs/Desktop/summerIntern/mvcom.png")





#density of floating16 point
import pylab as pl
import matplotlib.pyplot as plt
import numpy as np

significand_bits = 10
exponent_min = -14
exponent_max = 15
binwidth = 0.001

fp_numbers = []
for exp in range(exponent_min, exponent_max+1):
    for sbits in range(0, 2**significand_bits):
        significand = 1 + sbits/2**significand_bits 
        fp_numbers.append(significand * 2**exp)
        
fp_numbers = np.array(fp_numbers)
pl.gca().set_xscale("log",basex=2)
pl.hist(fp_numbers, bins=np.logspace(-14,15, num=6000,base=2.0))

pl.show()

bins = 2.0**(np.arange(-14,-12))
plt.xscale('log',basex=2)
plt.hist(fp_numbers,bins = bins)
#print(fp_numbers)

#pt.hist(fp_numbers, bins=np.arange(min(fp_numbers),max(fp_numbers)+binwidth*10,binwidth))
plt.hist(fp_numbers, bins=5)
#pt.plot(fp_numbers, np.ones_like(fp_numbers), "+")
#pt.semilogx(fp_numbers, np.ones_like(fp_numbers), "+")

#density of bfloating 16

significand_bits = 7
exponent_min = -254
exponent_max = 255

fp_numbers = []
for exp in range(exponent_min, exponent_max+1):
    for sbits in range(0, 2**significand_bits):
        significand = 1 + sbits/2**significand_bits 
        fp_numbers.append(significand * 2**exp)
        
fp_numbers = np.array(fp_numbers)
plt.hist(fp_numbers, bins=3)
bins = 2.0**(np.arange(-14,-12))
plt.xscale('log',basex=2)
plt.hist(fp_numbers,bins = bins)



"""week4_error propagation graph"""
"""sigmoid function"""
"""bf16 first layer"""

meanbf16_1 = (1.0299807108499846e-16, 1.0005838511168755e-14, 1.1950963490239624e-12, 1.1272752906135753e-10, 1.028557381663998e-08, 1.2085340173270424e-06, 0.00011512724105740743,0.010912770116108821, 1.1323429169072086, 110.88942918959286, 716.3451857369)
stdbf16_1= (3.9992527297248016e-17, 3.917957738351406e-15,4.622152331918473e-13, 4.193081432925387e-11,3.917261822082698e-09,4.5263828681121974e-07,4.778423331389008e-05,0.004230410979726093,0.47029042944475985,38.612740847788665,152.46862911433735)
"""fl16 first layer"""
meanfl16_1=(1.2482272087106862e-14,1.2488740880965015e-12, 1.330892930117689e-10, 1.321772672096623e-08, 4.4168879033876184e-08, 5.491942728772643e-08, 4.282116058123102e-06, 0.00042757416483919265, 0.0441565229746197,4.270095670297127, 27.566035988040795)
stdfl16_1=(3.724784049597544e-15,4.0055547388534926e-13, 3.8470904932610736e-11,4.293010184716444e-09,1.5668417318514176e-08,2.404282979048769e-08,2.4162727227632406e-06,0.00023474215791158054, 0.022167361347480152, 2.371666876257218,10.46048050260168)

"""bf16 second layer"""
meanbf16_2=(0.0,0.0,0.0,0.0,2.3422330044581477e-07, 2.3825434341364612e-05,0.0021745645971487466,0.00439872679541347,0.00030105471611022947, 0.0,0.0 ) 
stdbf16_2=(0.0,0.0,0.0,0.0,8.036966855951453e-08, 9.024487196123634e-06,0.0007595196858133972,0.0011619519866185241,0.0009980413127493016,0.0,0.0)

"""fl16 second layer"""
meanfl16_2=(0.0,0.0,0.0,0.0,2.3422330044581477e-07,2.3825434341364612e-05,0.0001962787806128421,0.00025461149640322654,8.45789909362793e-06,0.0,0.0 )
stdfl16_2=(0.0,0.0,0.0,0.0,8.036966855951453e-08,9.024487196123634e-06,8.2290130491055e-05,0.00013491679355738867,5.387530373723124e-05,0.0,0.0)

import matplotlib.pyplot as plt
import numpy as np
#import plotly.plotly as py
ax = plt.subplot(111)
ind = np.arange(len(meanbf16))
p1=ax.bar(ind-0.2, meanbf16_1,width=0.1,  color='indianred',align='center')
p2=ax.bar(ind-0.1, meanfl16_1,width=0.1, color='seagreen',align='center')
p3=ax.bar(ind+0.1, meanbf16_2,width=0.1,  color='r',align='center')
p4=ax.bar(ind+0.2, meanfl16_2,width=0.1, color='g',align='center')

plt.title("error propagation with Sigmoid function")
plt.xlabel("exponent of 2")
plt.ylabel("absolute error in difference with float32")
plt.xticks(ind, ('-24', '-20', '-17', '-14', '-10','-7','-4','-1','2','5','7'))
#plt.gca().set_xscale('log',basex=2)
plt.gca().set_yscale('log',basey=2)
plt.legend((p1[0], p2[0],p3[0],p4[0]), ('bf16_1', 'fl16_1','bf16_2','fl16_2'))
plt.gca().set_ylim(ymin=0)

plt.savefig("/home/uic-cs/Desktop/summerIntern/errorpropagtion_sigmoid.png")

"""tanh function"""
"""bf 16 first layer"""
meanbf16_1 = (1.0299807108499846e-16, 1.0005838511168755e-14, 1.1950963490239624e-12, 1.1272752906135753e-10, 1.028557381663998e-08, 1.2085340173270424e-06, 0.00011512724105740743,0.010912770116108821, 1.1323429169072086, 110.88942918959286, 716.3451857369)
stdbf16_1= (3.9992527297248016e-17, 3.917957738351406e-15,4.622152331918473e-13, 4.193081432925387e-11,3.917261822082698e-09,4.5263828681121974e-07,4.778423331389008e-05,0.004230410979726093,0.47029042944475985,38.612740847788665,152.46862911433735)
"""fl16 first layer"""
meanfl16_1=(1.2482272087106862e-14,1.2488740880965015e-12, 1.330892930117689e-10, 1.321772672096623e-08, 4.4168879033876184e-08, 5.491942728772643e-08, 4.282116058123102e-06, 0.00042757416483919265, 0.0441565229746197,4.270095670297127, 27.566035988040795)
stdfl16_1=(3.724784049597544e-15,4.0055547388534926e-13, 3.8470904932610736e-11,4.293010184716444e-09,1.5668417318514176e-08,2.404282979048769e-08,2.4162727227632406e-06,0.00023474215791158054, 0.022167361347480152, 2.371666876257218,10.46048050260168)

"""second layer"""
bf16_2 = (8.102488806613088e-17, 7.3018881758417e-15, 8.224916285693519e-13, 7.444364158835973e-11,7.527936976747143e-09, 8.51775467915933e-07, 0.00012549016129761694, 0.0064339907396884985,  3.8961172103881834e-05, 0.0,0.0)
stdbf16_2 = (3.1412070878967715e-17, 3.04247072960096e-15, 4.229485962997027e-13,  3.14386510582902e-11,3.2353735539951856e-09, 3.8381108620785915e-07, 4.838166762115465e-05,0.0016279501648136414, 0.0003876587677831131,0.0, 0.0)
fl16_2 = (9.474012029641775e-15,9.1109647509072e-13, 9.338361785304764e-11, 8.905742666592947e-09, 3.007440741459795e-08, 3.671959938297849e-08, 2.668721488540938e-06, 0.0002101469028925337, 1.0132789611816406e-07, 0.0,  0.0)
stdfl16_2 = (3.293896344681707e-15,3.0456286599750315e-13, 3.5678851549981425e-11, 2.7714612051698236e-09, 1.4221990229226526e-08, 2.2583851004979417e-08,1.823222463236468e-06,0.00010169720527229235, 1.0081998366601787e-06,0,0,0.0)

import matplotlib.pyplot as plt
import numpy as np
#import plotly.plotly as py
ax = plt.subplot(111)
ind = np.arange(len(meanbf16_1))
p1=ax.bar(ind-0.2, meanbf16_1,width=0.1,  color='indianred',align='center')
p2=ax.bar(ind-0.1, meanfl16_1,width=0.1, color='seagreen',align='center')
p3=ax.bar(ind+0.1, bf16_2,width=0.1,  color='r',align='center')
p4=ax.bar(ind+0.2, fl16_2,width=0.1, color='g',align='center')

plt.title("error propagation with Tanh function")
plt.xlabel("exponent of 2")
plt.ylabel("absolute error in difference with float32")
plt.xticks(ind, ('-24', '-20', '-17', '-14', '-10','-7','-4','-1','2','5','7'))
#plt.gca().set_xscale('log',basex=2)
plt.gca().set_yscale('log',basey=2)
plt.legend((p1[0], p2[0],p3[0],p4[0]), ('bf16_1', 'fl16_1','bf16_2','fl16_2'))
plt.savefig("/home/uic-cs/Desktop/summerIntern/errorpropagtion_tanh.png")


"""only second layer for tanh"""
bf16_2 = (8.102488806613088e-17, 7.3018881758417e-15, 8.224916285693519e-13, 7.444364158835973e-11,7.527936976747143e-09, 8.51775467915933e-07, 0.00012549016129761694, 0.0064339907396884985,  3.8961172103881834e-05, 0.0,0.0)
stdbf16_2 = (3.1412070878967715e-17, 3.04247072960096e-15, 4.229485962997027e-13,  3.14386510582902e-11,3.2353735539951856e-09, 3.8381108620785915e-07, 4.838166762115465e-05,0.0016279501648136414, 0.0003876587677831131,0.0, 0.0)
fl16_2 = (9.474012029641775e-15,9.1109647509072e-13, 9.338361785304764e-11, 8.905742666592947e-09, 3.007440741459795e-08, 3.671959938297849e-08, 2.668721488540938e-06, 0.0002101469028925337, 1.0132789611816406e-07, 0.0,  0.0)
stdfl16_2 = (3.293896344681707e-15,3.0456286599750315e-13, 3.5678851549981425e-11, 2.7714612051698236e-09, 1.4221990229226526e-08, 2.2583851004979417e-08,1.823222463236468e-06,0.00010169720527229235, 1.0081998366601787e-06,0,0,0.0)
import matplotlib.pyplot as plt
import numpy as np
#import plotly.plotly as py
ax = plt.subplot(111)
ind = np.arange(len(meanbf16_1))
p1=ax.bar(ind-0.1, bf16_2,width=0.2,  color='b',align='center')
p2=ax.bar(ind+0.1, fl16_2,width=0.2, color='g',align='center')
plt.title("error propagation with Tanh function")
plt.xlabel("exponent of 2")
plt.ylabel("absolute error in difference with float32")
plt.xticks(ind, ('-24', '-20', '-17', '-14', '-10','-7','-4','-1','2','5','7'))
#plt.gca().set_xscale('log',basex=2)
plt.gca().set_yscale('log',basey=2)
plt.legend((p1[0], p2[0]), ('bf16', 'fl16'))
plt.savefig("/home/uic-cs/Desktop/summerIntern/errorpropagtion_tanhonlylayer2.png")



"""only second layer for sigmoid"""
"""bf16 second layer"""
bf16_2=(0.0,0.0,0.0,0.0,2.3422330044581477e-07, 2.3825434341364612e-05,0.0021745645971487466,0.00439872679541347,0.00030105471611022947, 0.0,0.0 ) 
stdbf16_2=(0.0,0.0,0.0,0.0,8.036966855951453e-08, 9.024487196123634e-06,0.0007595196858133972,0.0011619519866185241,0.0009980413127493016,0.0,0.0)
"""fl16 second layer"""
fl16_2=(0.0,0.0,0.0,0.0,2.3422330044581477e-07,2.3825434341364612e-05,0.0001962787806128421,0.00025461149640322654,8.45789909362793e-06,0.0,0.0 )
stdfl16_2=(0.0,0.0,0.0,0.0,8.036966855951453e-08,9.024487196123634e-06,8.2290130491055e-05,0.00013491679355738867,5.387530373723124e-05,0.0,0.0)

import matplotlib.pyplot as plt
import numpy as np
#import plotly.plotly as py
ax = plt.subplot(111)
ind = np.arange(len(meanbf16_1))
p1=ax.bar(ind-0.1, bf16_2,width=0.2,  color='b',align='center')
p2=ax.bar(ind+0.1, fl16_2,width=0.2, color='g',align='center')
plt.title("error propagation with sigmoid function")
plt.xlabel("exponent of 2")
plt.ylabel("absolute error in difference with float32")
plt.xticks(ind, ('-24', '-20', '-17', '-14', '-10','-7','-4','-1','2','5','7'))
#plt.gca().set_xscale('log',basex=2)
plt.gca().set_yscale('log',basey=2)
plt.legend((p1[0], p2[0]), ('bf16', 'fl16'))
plt.savefig("/home/uic-cs/Desktop/summerIntern/errorpropagtion_sigmoidonlylayer2.png")












