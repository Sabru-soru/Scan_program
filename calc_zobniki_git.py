"""
Created  April 2019
@author: Urbas
"""
# =============================================================================
# Zaenkrat uporabimo modul, da izračunamo razdelni krog.
# Logika se podre, če so scani orientirani drugače
# Ko bomo uredili model in ga razrezali je za preverit kolikšen del se zanemari pri flank profile
# Flank profil določen po cilindru.
# Kot alpha je določen ročno na 20
# cca 1600 za report *1000


"POMEMBNO!!!!!!!"
#  za nave zobnike 2021        if fi_cut_calc>88:                          #this is 90 by default
# zaenkrat so fiksirane DIN vrednosti ker ne računa vredu
# temporary
# =============================================================================

import numpy as np 
import matplotlib.pyplot as plt
import math
#import itertools
import shapely.geometry as shp
import os
from prettytable import PrettyTable
from pandas import DataFrame
import pandas as pd
import openpyxl
import matplotlib.font_manager as font_manager

# We import the files in lines ~ 40 and 700
# Saved as 'Plane Z +0.000 mm.asc' and 'Cylinder 1 +0.000 mm.asc'
# line ~700 determines what flank side to display
# Line ~1000 determines what tooth involute evaluation to display
# line ~1700 is the file for MBD
# Line 850 & 1200 for what graphs to display
#%%
"Import of the sections."

#folder='Novi_scani_10_12_2020\\obdelava\\1_10\\1-10_3_geometric_alignment'
#folder='Novi_scani_10_12_2020\\obdelava\\4_10'
#folder='Novi_scani_10_12_2020\\obdelava\\CAD_obdelava'
#folder='C:\\Users\\urosu\\Dropbox\\MR\Konference & clanki\\Boljsa_poravnava\\zmodelirani_odstopki\\nasa_poravnava\\dolocanje_parametrov'
#folder='C:\\Users\\urosu\\Dropbox\\MR\\Konference & clanki\\Boljsa_poravnava\\obdelava podatkov\\vrednotenje_poravnav\\Tecaform1\\nasa_poravnava\\14_5_ICP_poizkus\\za_sliko_orientacija'
#folder='C:\\Users\\urosu\\Dropbox\\MR\\Konference & clanki\\Boljsa_poravnava\\obdelava podatkov\\vrednotenje_poravnav\\Tecaform1\\referencna_poravnava_29_3_2021\\za_sliko_python'
folder='C:\\Users\\urosu\\Dropbox\\MR\Konference & clanki\\Boljsa_poravnava\\obdelava podatkov\\zmodelirani_odstopki\\zamaknjena_luknja\\python_GOM'
X=[]
Y=[]

owd=os.getcwd()
os.chdir(folder)
gearwd=os.getcwd()
file = open('Plane Z +0.000 mm.asc',"r")    #planar section
os.chdir(owd)

for line in file:
    x, y, z = line.split()
    X.append(float(x))                      #need to assign float
    Y.append(float(y))
     
# =============================================================================
# plt.plot(X,Y,marker='o', markersize=1, linestyle=' ')
# plt.plot(0, 0,'ro',label='Origin C.S.')
# plt.axis('equal')
# plt.title('Gear')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid()
# plt.show()
# =============================================================================
    
#=================================
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
"INPUT DATA"
alpha=20 # used for r_b in involute evaluation
k_eval_teeth=2 #used for parameter F_pk, 2 or 3
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#=================================

#%% 
#"Identify inner circle."
r=[]
for i in range(len(X)):
    r.append(math.sqrt(X[i]**2 + Y[i]**2))

# =============================================================================
# plt.plot(r,range(len(r)), marker='o', markersize=1, linestyle=' ')
# plt.show()
# =============================================================================

r_cutoff=(np.max(r)+np.min(r))*1/2 #2/3        #define the radius for the cutoff
#r_cutoff=np.min(r)*1.3
#r_cutoff=4
circleNindex=[]
#for i in r:
#    if i <r_cutoff:
#        #print(r.index(i))
#        #print(i)
#        circleNindex.append(r.index(i)) #a set of indexses for the inner circle

circleNindex=[]
for i in range(len(r)):
    if r[i] <r_cutoff:
        circleNindex.append(i) #a set of indexes for the inner circle

"Remove the inner circle values"
Xinner=[]
Yinner=[]
c=0
for i in circleNindex:
    Xinner.append(X[i-c])
    Yinner.append(Y[i-c])
    X.pop(i-c)
    Y.pop(i-c)
    c=c+1

plt.plot(X,Y,marker='o', markerfacecolor='black', markersize=1, linestyle=' ')
plt.plot(Xinner,Yinner,marker='s', markerfacecolor='g', markersize=1, linestyle=' ')
plt.plot(0, 0,'ro',label='Origin C.S.')
plt.axis('equal')
plt.title('Gear')
plt.xlabel('x [mm]')
plt.ylabel('y [mm]')
plt.grid()
plt.tight_layout()
os.chdir(gearwd)
plt.savefig("General.png", dpi = 150)
os.chdir(owd)
plt.show()

"Circular coordinate system"
r=[]
fi=[]
angle=0
for i in range(len(X)):
    r.append(math.sqrt(X[i]**2 + Y[i]**2))
    angle=math.atan2(Y[i],X[i])*(180/np.pi) #angle calculation, return 0<fi<360
    if angle<0:
        angle=angle+360
    fi.append(angle)

"Validation"
fi_val=[]
for i in range(len(fi)):
    fi_val.append(fi[i]*np.pi/180)

X_val=[]
Y_val=[]
X_val=r*np.cos(fi_val)    #need angle in radians
Y_val=r*np.sin(fi_val)    #need angle in radians
# =============================================================================
# plt.plot(X_val,Y_val,marker='o', markerfacecolor='black', markersize=1, linestyle=' ')
# plt.plot(0, 0,'ro',label='Origin C.S.')
# plt.axis('equal')
# plt.title('Gear_validation')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid()
# plt.show()
# =============================================================================

# a lot of calculation. Need to calculate with original indexes and values.
#%%
"Sort the points by angle fi"
fi_r_sorted=sorted(zip(fi,r))
r_sorted=[r for fi, r in fi_r_sorted]
fi_sorted=sorted(fi)

fi_sorted_val=[]
for i in range(len(fi_sorted)):
    fi_sorted_val.append(fi_sorted[i]*np.pi/180)
X_sorted_val=[]
Y_sorted_val=[]
X_sorted_val=r_sorted*np.cos(fi_sorted_val)    #need angle in radians
Y_sorted_val=r_sorted*np.sin(fi_sorted_val)    #need angle in radians
# =============================================================================
# plt.plot(X_sorted_val,Y_sorted_val,marker='o', markerfacecolor='black', markersize=1, linestyle=' ')
# plt.plot(0, 0,'ro',label='Origin C.S.')
# plt.axis('equal')
# plt.title('Gear_validation_sorted')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid()
# plt.show()
# =============================================================================

#%% 
"Calculate the number of teeth"
# =============================================================================
# plt.plot(r_sorted)
# xr_sorted=range(len(r_sorted))
# diff1=[]
# for i in range(1,len(r_sorted)):
#     diff1.append((r_sorted[i]-r_sorted[i-1])/np.abs((xr_sorted[i]-xr_sorted[i-1])/50)) #calculating the first diferential
# plt.plot(diff1)
# plt.show()
# 
# for i in range(1,len(diff1)-1): #removing outliers, if its negative in a pool of positive data and vice versa
#     if diff1[i]<0 and diff1[i+1]>0 and diff1[i-1]>0:
#         diff1[i]=diff1[i]*(-1)
#     if diff1[i]>0 and diff1[i+1]<0 and diff1[i-1]<0:
#         diff1[i]=diff1[i]*(-1)
# 
# N_teeth=int(len(list(itertools.groupby(diff1, lambda diff1: diff1 > 0)))/2) #determines the number of times the values cross from minus to plus, devide by 2 because we get maximums and minimums
# 
# =============================================================================

#Finds the transition near the root, to determine the number of teeth
tooth=0
r_avg=(np.max(r_sorted)+np.min(r_sorted))/2
for i in range(2,len(r_sorted)-1):
    if r_sorted[i]<r_avg and r_sorted[i+1]<r_avg and r_sorted[i-1]>r_avg and r_sorted[i-2]>r_avg:
        tooth+=1
N_teeth=tooth
#N_teeth=39

print('The number of teeth is:',N_teeth)
fi_range=360/N_teeth

"Divide the teeth"
fi_divided=[[] for x in range(N_teeth)]
r_divided=[[] for x in range(N_teeth)]
for i in range(len(fi_sorted)):
    if fi_sorted[i]<fi_range/2 or fi_sorted[i]>(360-(fi_range/2)):
        fi_divided[0].append(fi_sorted[i])
        r_divided[0].append(r_sorted[i])
    else:
        for j in range(N_teeth):
            if fi_sorted[i]>fi_range/2+(fi_range*(j-1)) and fi_sorted[i]<(360-fi_range/2-fi_range*(N_teeth-1-j)):
                fi_divided[j].append(fi_sorted[i])
                r_divided[j].append(r_sorted[i])
                
# "Plot to check"
fi_divided_val=[[] for x in range(N_teeth)]
X_divided_val=[[] for x in range(N_teeth)]
Y_divided_val=[[] for x in range(N_teeth)]
for i in range(len(fi_divided)):
    for j in range(len(fi_divided[i])):
        fi_divided_val[i].append(fi_divided[i][j]*np.pi/180)                     #turn to radians for plotting                
        X_divided_val[i].append(r_divided[i][j]*np.cos(fi_divided_val[i][j]))    #need angle in radians
        Y_divided_val[i].append(r_divided[i][j]*np.sin(fi_divided_val[i][j]))    #need angle in radians

# #=============================================================================
# for i in range(len(X_divided_val)):           
#     plt.plot(X_divided_val[i],Y_divided_val[i],marker='o', markerfacecolor='black', markersize=1, linestyle=' ')
# plt.plot(0, 0,'ro',label='Origin C.S.')
# plt.axis('equal')
# plt.title('Gear_validation_divided')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid()
# plt.show()
#=============================================================================

#%%
#======================================================================================================================= 
"Calculate the pitch circle and module"

estimated_r_pc=np.mean(r_sorted)
estimated_r_pc_list=[]
estimated_fi_pc_list=[]
for i in range(N_teeth): #on the right flank
    for j in range(1,len(r_divided[i])):
        if r_divided[i][j-1]<=estimated_r_pc and r_divided[i][j]>=estimated_r_pc:
            estimated_r_pc_list.append((r_divided[i][j-1]+r_divided[i][j-1])/2)
            estimated_fi_pc_list.append((fi_divided[i][j-1]+fi_divided[i][j-1])/2)

estimated_fi_pc_diff=[]
for i in range(1,len(estimated_fi_pc_list)):
    estimated_fi_pc_diff.append(estimated_fi_pc_list[i]-estimated_fi_pc_list[i-1]) 
estimated_fi_pc_diff.append(estimated_fi_pc_list[0]-estimated_fi_pc_list[-1])
if estimated_fi_pc_diff[0]<0:
    estimated_fi_pc_diff[0]=estimated_fi_pc_list[1]+np.abs(estimated_fi_pc_list[0]-360)

estimated_r_pc_list_mean=np.mean(estimated_r_pc_list)
estimated_p_rc_diff=[]
for i in range(len(estimated_fi_pc_diff)):
    estimated_p_rc_diff.append(np.radians(estimated_fi_pc_diff[i])*estimated_r_pc_list_mean)
estimated_p_rc_diff_mean=np.mean(np.abs(estimated_p_rc_diff)) #calculate the pitch to then determine the module
    
module=round(estimated_p_rc_diff_mean/np.pi,1) #We calculated the module, now we round it to the first digit
print('The module is:',module)
#=======================================================================================================================
module=1
pitch_circle=module*N_teeth
#pitch_circle=38.482 #temporary
r_pc=pitch_circle/2         #pitch circle radius

# #%%

# f_pt_left=0
# f_pt_right=0
# Fpk_max_diff_left=0
# Fpk_max_diff_right=0
# F_p_left=0
# F_p_right=0
#%%
"Determine the points on the pitch circle for f_pt and F_p"

r_pt_right=[]
fi_pt_right=[]
r_mid=0
fi_mid=0
r_mid2_down=0
fi_mid2_down=0
r_mid2_up=0
fi_mid2_up=0
#This was a double interpolation, we now use the sistem of equations to determine the points more accurately
#for i in range(N_teeth): 
#    for j in range(1,len(r_divided[i])):
#        if r_divided[i][j-1]<=r_pc and r_divided[i][j]>=r_pc:
#            r_mid=(r_divided[i][j]+r_divided[i][j-1])/2
#            fi_mid=(fi_divided[i][j]+fi_divided[i][j-1])/2
#            #
#            r_mid2_down=(r_mid+r_divided[i][j-1])/2
#            fi_mid2_down=(fi_mid+fi_divided[i][j-1])/2
#            r_mid2_up=(r_mid+r_divided[i][j])/2
#            fi_mid2_up=(fi_mid+fi_divided[i][j])/2
#            #
#            if r_pc>=r_divided[i][j-1] and r_pc<=r_mid2_down:
#                r_pt_right.append((r_mid2_down+r_divided[i][j-1])/2)
#                fi_pt_right.append((fi_mid2_down+fi_divided[i][j-1])/2)
#            elif r_pc<=r_mid and r_pc>=r_mid2_down:
#                r_pt_right.append((r_mid2_down+r_mid)/2)
#                fi_pt_right.append((fi_mid2_down+fi_mid)/2) 
#            elif r_pc>=r_mid and r_pc<=r_mid2_up:
#                r_pt_right.append((r_mid2_up+r_mid)/2)
#                fi_pt_right.append((fi_mid2_up+fi_mid)/2)
#            elif r_pc>=r_mid2_up and r_pc<=r_divided[i][j]:
#                r_pt_right.append((r_mid2_up+r_divided[i][j])/2)
#                fi_pt_right.append((fi_mid2_up+fi_divided[i][j])/2)

#k_slope=0  
#n_line=0              
#for i in range(N_teeth):
#    for j in range(1,len(r_divided[i])):
#        if r_divided[i][j-1]<=r_pc and r_divided[i][j]>=r_pc:
#            k_slope=(fi_divided[i][j]-fi_divided[i][j-1])/(r_divided[i][j]-r_divided[i][j-1])
#            n_line=fi_divided[i][j-1]-k_slope*fi_divided[i][j-1]
#            fi_to_append=k_slope*r_pc+n_line
#            r_pt_right.append(r_pc)
#            fi_pt_right.append(fi_to_append)

import sympy
"For the right flank"   
X_up=0
Y_up=0
X_down=0
Y_down=0
k_slope=0  
n_line=0
angle_in_deg=0
X_solve=0
Y_solve=0
r_pt_right=[]
fi_pt_right=[]
aa=0           
for i in range(N_teeth):
    for j in range(1,len(r_divided[i])):
        if r_divided[i][j-1]<=r_pc and r_divided[i][j]>=r_pc:
            X_up=(r_divided[i][j]*np.cos(np.radians(fi_divided[i][j])))          #need angle in radians
            Y_up=(r_divided[i][j]*np.sin(np.radians(fi_divided[i][j])))
            X_down=(r_divided[i][j-1]*np.cos(np.radians(fi_divided[i][j-1])))          #need angle in radians
            Y_down=(r_divided[i][j-1]*np.sin(np.radians(fi_divided[i][j-1])))
            #
            k_slope=(Y_up-Y_down)/(X_up-X_down)
            n_line=Y_up-k_slope*X_up
            x, y = sympy.symbols('x, y')
            aa=sympy.solve([(x**2+y**2)**(1/2)-r_pc, k_slope*x+n_line-y], x, y)
            #print(aa)
            #if (i>=0 and i<=4) or (i>=15 and i<=19):  #we get two sets of solutions
            if i>0 and i<=(math.ceil(N_teeth/4)-1):                                      #first quadrant
                if aa[0][0]>=0 and aa[0][1]>=0:
                  X_solve=aa[0][0]
                  Y_solve=aa[0][1]
                else:
                  X_solve=aa[1][0]
                  Y_solve=aa[1][1]
            if i>=(math.ceil(N_teeth/4)+1) and i<=(math.ceil(N_teeth/2)-1):                                      #second quadrant
                if aa[0][0]<=0 and aa[0][1]>=0:
                  X_solve=aa[0][0]
                  Y_solve=aa[0][1]
                else:
                  X_solve=aa[1][0]
                  Y_solve=aa[1][1]
            if i>=(math.ceil(N_teeth/2)+1) and i<=(math.ceil(N_teeth*(3/4))-1):                                      #third quadrant
                if aa[0][0]<=0 and aa[0][1]<=0:
                  X_solve=aa[0][0]
                  Y_solve=aa[0][1]
                else:
                  X_solve=aa[1][0]
                  Y_solve=aa[1][1]
            if i>=math.ceil(N_teeth*(3/4))+1 and i<=(N_teeth-1):                                      #fourth quadrant
                if aa[0][0]>=0 and aa[0][1]<=0:
                  X_solve=aa[0][0]
                  Y_solve=aa[0][1]
                else:
                  X_solve=aa[1][0]
                  Y_solve=aa[1][1]
            if i==0:                                      #tooth zero
                if aa[0][0]>=0 and aa[0][1]<=0:
                  X_solve=aa[0][0]
                  Y_solve=aa[0][1]
                else:
                  X_solve=aa[1][0]
                  Y_solve=aa[1][1]
            if i>=math.ceil(N_teeth/4)-1 and i<=math.ceil(N_teeth/4)+1: #90°
                if aa[0][1]>0:
                  X_solve=aa[0][0]
                  Y_solve=aa[0][1]
                else:
                  X_solve=aa[1][0]
                  Y_solve=aa[1][1]
            if i>=math.ceil(N_teeth/2)-1 and i<=math.ceil(N_teeth/2)+1: #180°
                if aa[0][0]<0:
                  X_solve=aa[0][0]
                  Y_solve=aa[0][1]
                else:
                  X_solve=aa[1][0]
                  Y_solve=aa[1][1]    
            if i>=math.ceil(N_teeth*3/4)-1 and i<=math.ceil(N_teeth*3/4)+1: #270°
                if aa[0][1]<0:
                  X_solve=aa[0][0]
                  Y_solve=aa[0][1]
                else:
                  X_solve=aa[1][0]
                  Y_solve=aa[1][1]        
            r_pt_right.append(math.sqrt(X_solve**2 + Y_solve**2))
            angle_in_deg=math.atan2(Y_solve,X_solve)*(180/np.pi)
            if angle_in_deg<0:
                angle_in_deg=angle_in_deg+360
            fi_pt_right.append(angle_in_deg)
            aa=0            
 

"For the left flank"        
X_up=0
Y_up=0
X_down=0
Y_down=0
k_slope=0  
n_line=0
angle_in_deg=0
X_solve=0
Y_solve=0
r_pt_left=[]
fi_pt_left=[]
aa=0
for i in range(N_teeth):
    for j in range(1,len(r_divided[i])):
        if r_divided[i][j]<=r_pc and r_divided[i][j-1]>=r_pc:
            X_up=(r_divided[i][j-1]*np.cos(np.radians(fi_divided[i][j-1])))          #need angle in radians
            Y_up=(r_divided[i][j-1]*np.sin(np.radians(fi_divided[i][j-1])))
            X_down=(r_divided[i][j]*np.cos(np.radians(fi_divided[i][j])))          #need angle in radians
            Y_down=(r_divided[i][j]*np.sin(np.radians(fi_divided[i][j])))
            #
            k_slope=(Y_up-Y_down)/(X_up-X_down)
            n_line=Y_up-k_slope*X_up
            x, y = sympy.symbols('x, y')
            aa=sympy.solve([(x**2+y**2)**(1/2)-r_pc, k_slope*x+n_line-y], x, y)
            #print(aa)
            #if (i>=0 and i<=5) or (i>=16 and i<=19):  #we get two sets of solutions 
            if i>0 and i<=(math.ceil(N_teeth/4)-1):                                      #first quadrant
                if aa[0][0]>=0 and aa[0][1]>=0:
                  X_solve=aa[0][0]
                  Y_solve=aa[0][1]
                else:
                  X_solve=aa[1][0]
                  Y_solve=aa[1][1]
            if i>=(math.ceil(N_teeth/4)+1) and i<=(math.ceil(N_teeth/2)-1):                                      #second quadrant
                if aa[0][0]<=0 and aa[0][1]>=0:
                  X_solve=aa[0][0]
                  Y_solve=aa[0][1]
                else:
                  X_solve=aa[1][0]
                  Y_solve=aa[1][1]
            if i>=(math.ceil(N_teeth/2)+1) and i<=(math.ceil(N_teeth*(3/4))-1):                                      #third quadrant
                if aa[0][0]<=0 and aa[0][1]<=0:
                  X_solve=aa[0][0]
                  Y_solve=aa[0][1]
                else:
                  X_solve=aa[1][0]
                  Y_solve=aa[1][1]
            if i>=math.ceil(N_teeth*(3/4))+1 and i<=(N_teeth-1):                                      #fourth quadrant
                if aa[0][0]>=0 and aa[0][1]<=0:
                  X_solve=aa[0][0]
                  Y_solve=aa[0][1]
                else:
                  X_solve=aa[1][0]
                  Y_solve=aa[1][1]
            if i==0:                                      #tooth zero
                if aa[0][0]>=0:
                  X_solve=aa[0][0]
                  Y_solve=aa[0][1]
                else:
                  X_solve=aa[1][0]
                  Y_solve=aa[1][1]
            if i>=math.ceil(N_teeth/4)-1 and i<=math.ceil(N_teeth/4)+1: #90°
                if aa[0][1]>0:
                  X_solve=aa[0][0]
                  Y_solve=aa[0][1]
                else:
                  X_solve=aa[1][0]
                  Y_solve=aa[1][1]
            if i>=math.ceil(N_teeth/2)-1 and i<=math.ceil(N_teeth/2)+1: #180°
                if aa[0][0]<0:
                  X_solve=aa[0][0]
                  Y_solve=aa[0][1]
                else:
                  X_solve=aa[1][0]
                  Y_solve=aa[1][1]    
            if i>=math.ceil(N_teeth*3/4)-1 and i<=math.ceil(N_teeth*3/4)+1: #270°
                if aa[0][1]<0:
                  X_solve=aa[0][0]
                  Y_solve=aa[0][1]
                else:
                  X_solve=aa[1][0]
                  Y_solve=aa[1][1]        
            r_pt_left.append(math.sqrt(X_solve**2 + Y_solve**2))
            angle_in_deg=math.atan2(Y_solve,X_solve)*(180/np.pi)
            if angle_in_deg<0:
                angle_in_deg=angle_in_deg+360
            fi_pt_left.append(angle_in_deg)
            aa=0            



"Evaluation of the points"
X_pt_eval=[]
Y_pt_eval=[]
for i in range(len(fi_pt_right)):
    X_pt_eval.append(r_pt_right[i]*np.cos(np.radians(fi_pt_right[i])))          #need angle in radians
    Y_pt_eval.append(r_pt_right[i]*np.sin(np.radians(fi_pt_right[i])))          #need angle in radians
    X_pt_eval.append(r_pt_left[i]*np.cos(np.radians(fi_pt_left[i])))            #need angle in radians
    Y_pt_eval.append(r_pt_left[i]*np.sin(np.radians(fi_pt_left[i])))

for i in range(len(X_divided_val)):           
    plt.plot(X_divided_val[i],Y_divided_val[i],marker='o', markerfacecolor='black', markersize=1, linestyle=' ')
plt.plot(0, 0,'ro',label='Origin C.S.')
plt.plot(X_pt_eval,Y_pt_eval,marker='x',markersize=5,linestyle=' ')
plt.axis('equal')
plt.title('Gear_validation_divided')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()

## Plot for report
axis_font = {'fontname':'Times new Roman', 'size':'16'}

circle_i=np.arange(np.pi/2.9,np.pi/1.7,0.1)
x_for_circle_i=r_pc*np.cos(circle_i)
y_for_circle_i=r_pc*np.sin(circle_i)

f = plt.figure()
plt.plot(x_for_circle_i,y_for_circle_i, color='green', linestyle='-.',label="Reference circle")
plt.plot(X_divided_val[4],Y_divided_val[4], color='black', linestyle='-')#, marker='o', markersize=1)
plt.plot(X_divided_val[5],Y_divided_val[5], color='black', linestyle='-')#, marker='o', markersize=1)
plt.plot(X_pt_eval[9],Y_pt_eval[9],marker='o', color='red', markersize=7,linestyle=' ', label="Points for pitch control")
plt.plot(X_pt_eval[11],Y_pt_eval[11],marker='o', color='red', markersize=7,linestyle=' ')
plt.plot()
plt.axis('equal')
#plt.title('Pitch control',**axis_font)
plt.xlabel('$x$ [mm]',**axis_font)
plt.ylabel('$y$ [mm]',**axis_font)
plt.legend(prop={'family':'Times new Roman', 'size':16})
plt.grid()
plt.tight_layout()
axes = plt.gca()
axes.set_xlim([-2,4])
axes.set_ylim([8,12])
os.chdir(gearwd)
#plt.savefig("pitch control report.png", dpi = 200)
f.savefig("pitch_control.pdf", bbox_inches='tight')
os.chdir(owd)
plt.show()


##
##
#circle_i=np.arange(np.pi/1.92,np.pi/1.88,0.01)
#x_for_circle_i=r_pc*np.cos(circle_i)
#y_for_circle_i=r_pc*np.sin(circle_i)
#plt.plot(x_for_circle_i,y_for_circle_i, color='orange', linestyle='-.',label="Pitch circle")
#plt.plot([X_divided_val[5][170],X_divided_val[5][171]],[Y_divided_val[5][170],Y_divided_val[5][171]],linestyle='-')
#
#for i in range(166,175):  
#    plt.plot(X_divided_val[5][i],Y_divided_val[5][i], color='black', marker='o', markersize=3) #, marker='o', markersize=1)
#
#plt.plot(X_pt_eval[11],Y_pt_eval[11],marker='o', color='red', markersize=7,linestyle=' ', label="Points for pitch control")
#plt.plot()
#plt.axis('equal')
##plt.title('Pitch control',**axis_font)
#plt.xlabel('$x$ [mm]',**axis_font)
#plt.ylabel('$y$ [mm]',**axis_font)
##plt.legend(prop={'family':'Times new Roman', 'size':16})
#plt.grid()
#plt.tight_layout()
#os.chdir(gearwd)
#plt.savefig("pitch control report2.png", dpi = 150)
#os.chdir(owd)
#plt.show()

#%%
"Calculations of the pitch control for the left flank"
axis_font = {'fontname':'Times new Roman', 'size':'16'}

fi_pt_left_diff=[]
for i in range(1,len(fi_pt_left)):
    fi_pt_left_diff.append(fi_pt_left[i]-fi_pt_left[i-1])                       #calculations are done counterclockwise, 1,2,3 ...
fi_pt_left_diff.append(360-fi_pt_left[-1]+fi_pt_left[0])                        #add the last difference between the last and the first

f_pt_left=[]
for i in range(len(fi_pt_left_diff)):
    f_pt_left.append(np.radians(fi_pt_left_diff[i]-fi_range)*r_pc*1000)         #in micrometers

max_index_fpt=np.argmax(np.abs(f_pt_left))
f = plt.figure()
bars = plt.bar(range(1,len(f_pt_left)+1),f_pt_left)

plt.xlabel('N Division',**axis_font)
plt.ylabel('$f_{pt}$ [${\mu}m$]',**axis_font)
plt.title('$f_{pt}$ deviation, left flank',**axis_font) #counterclockwise
plt.xticks(rotation=90)
plt.grid()
plt.xlim(0.5, len(f_pt_left)+0.5)
plt.xticks(range(1, len(f_pt_left)+1))
plt.tight_layout()
yval = np.round(bars[max_index_fpt].get_height(),1)
plt.text(bars[max_index_fpt].get_x() + 1, yval, yval,**axis_font)
os.chdir(gearwd)
plt.savefig("f_pt_left.png", dpi = 200)
os.chdir(owd)
plt.show()
f_pt_max_left=np.max(np.abs(f_pt_left))
print('f_pt_max_left =',round(f_pt_max_left,1),'um')
f.savefig("pitch deviation.pdf", bbox_inches='tight')

F_pk_left=[]
F_pk_left.append(f_pt_left[0])
for i in range(1,len(f_pt_left)):
    F_pk_left.append(f_pt_left[i]+F_pk_left[i-1])

plt.bar(range(1,len(F_pk_left)+1),F_pk_left)
plt.xlabel('N Division',**axis_font)
plt.ylabel('$F_{pk}$ [${\mu}m$]',**axis_font)
plt.title('$F_{pk}$ deviation, left flank, counterclockwise',**axis_font)
plt.xticks(rotation=90)
plt.grid()
plt.xlim(0.5, len(F_pk_left)+0.5)
plt.xticks(range(1, len(F_pk_left)+1))
plt.tight_layout()
os.chdir(gearwd)
plt.savefig("F_pk_left.png", dpi = 200)
os.chdir(owd)
plt.show()

F_p_left=np.max(F_pk_left)+np.abs(np.min(F_pk_left))
print('F_p_left =',round(F_p_left,1),'um')


#==============================================================================
#F_pk_left calculations
F_pk_left_forcalc = F_pk_left[:-1]
F_pk_left_foreval=[]
F_pk_left_1 = F_pk_left_forcalc[:len(F_pk_left_forcalc)//2] #first half
F_pk_left_2 = F_pk_left_forcalc[len(F_pk_left_forcalc)//2:] #second half
F_pk_left_foreval=F_pk_left_2+F_pk_left_1 #another list, so we can evaluate from the back too

diff=0
Fpk_max_diff_left=0
for i in range(len(F_pk_left_forcalc)-2):
    diff=np.abs(F_pk_left_forcalc[i+1]-F_pk_left_forcalc[i])
    if diff>Fpk_max_diff_left:
        Fpk_max_diff_left=diff
for i in range(len(F_pk_left_forcalc)-2):
    diff=np.abs(F_pk_left_forcalc[i+k_eval_teeth-1]-F_pk_left_forcalc[i])
    if diff>Fpk_max_diff_left:
        Fpk_max_diff_left=diff    

for i in range(len(F_pk_left_foreval)-2):
    diff=np.abs(F_pk_left_foreval[i+1]-F_pk_left_foreval[i])
    if diff>Fpk_max_diff_left:
        Fpk_max_diff_left=diff
for i in range(len(F_pk_left_foreval)-2):
    diff=np.abs(F_pk_left_foreval[i+k_eval_teeth-1]-F_pk_left_foreval[i])
    if diff>Fpk_max_diff_left:
        Fpk_max_diff_left=diff   

print("Fp_",k_eval_teeth,"_left is:",Fpk_max_diff_left)
        
# =============================================================================
"Calculations for the right flank"
# =============================================================================

fi_pt_right_diff=[]
for i in range(1,len(fi_pt_right)):
    fi_pt_right_diff.append(fi_pt_right[i]-fi_pt_right[i-1]) #calculations are done counterclockwise, 1,2,3 ...
fi_pt_right_diff.append(fi_pt_right[0]-fi_pt_right[-1]) 
if fi_pt_right_diff[0]<0:
    fi_pt_right_diff[0]=fi_pt_right[1]+np.abs(fi_pt_right[0]-360)
    
f_pt_right=[]
for i in range(len(fi_pt_right_diff)):
    f_pt_right.append(np.radians(fi_pt_right_diff[i]-fi_range)*r_pc*1000) #in micrometers

plt.bar(range(1,len(f_pt_right)+1),f_pt_right)
plt.xlabel('N Division',**axis_font)
plt.ylabel('$f_{pt}$ [${\mu}m$]',**axis_font)
plt.title('$f_{pt}$ deviation, right flank, counterclockwise',**axis_font)
plt.grid()
plt.xticks(rotation=90)
plt.xlim(0.5, len(f_pt_right)+0.5)
plt.xticks(range(1, len(f_pt_right)+1))
plt.tight_layout()
os.chdir(gearwd)
plt.savefig("f_pt_right.png", dpi = 200)
os.chdir(owd)
plt.show()
f_pt_max_right=np.max(np.abs(f_pt_right))
print('f_pt_max_right =',round(f_pt_max_right,1),'um')

F_pk_right=[]
F_pk_right.append(f_pt_right[0])
for i in range(1,len(f_pt_right)):
    F_pk_right.append(f_pt_right[i]+F_pk_right[i-1])

plt.bar(range(1,len(F_pk_right)+1),F_pk_right)
plt.xlabel('N Division',**axis_font)
plt.ylabel('$F_{pk}$ [${\mu}m$]',**axis_font)
plt.title('$F_{pk}$ deviation, right flank, counterclockwise',**axis_font)
plt.grid()
plt.xticks(rotation=90)
plt.xlim(0.5, len(F_pk_right)+0.5)
plt.xticks(range(1, len(F_pk_right)+1))
plt.tight_layout()
os.chdir(gearwd)
plt.savefig("F_pk_right.png", dpi = 200)
os.chdir(owd)
plt.show()

F_p_right=np.max(F_pk_right)+np.abs(np.min(F_pk_right))
print('F_p_right =',round(F_p_right,1),'um')

#==============================================================================
#F_pk_right calculations
F_pk_right_forcalc = F_pk_right[:-1]
F_pk_right_foreval=[]
F_pk_right_1 = F_pk_right_forcalc[:len(F_pk_right_forcalc)//2] #first half
F_pk_right_2 = F_pk_right_forcalc[len(F_pk_right_forcalc)//2:] #second half
F_pk_right_foreval=F_pk_right_2+F_pk_right_1 #another list, so we can evaluate from the back too

diff=0
Fpk_max_diff_right=0
for i in range(len(F_pk_right_forcalc)-2):
    diff=np.abs(F_pk_right_forcalc[i+1]-F_pk_right_forcalc[i])
    if diff>Fpk_max_diff_right:
        Fpk_max_diff_right=diff
for i in range(len(F_pk_right_forcalc)-2):
    diff=np.abs(F_pk_right_forcalc[i+k_eval_teeth-1]-F_pk_right_forcalc[i])
    if diff>Fpk_max_diff_right:
        Fpk_max_diff_right=diff    

for i in range(len(F_pk_right_foreval)-2):
    diff=np.abs(F_pk_right_foreval[i+1]-F_pk_right_foreval[i])
    if diff>Fpk_max_diff_right:
        Fpk_max_diff_right=diff
for i in range(len(F_pk_right_foreval)-2):
    diff=np.abs(F_pk_right_foreval[i+k_eval_teeth-1]-F_pk_right_foreval[i])
    if diff>Fpk_max_diff_right:
        Fpk_max_diff_right=diff  

print("Fp_",k_eval_teeth,"_right is:",Fpk_max_diff_right)


# for image 11_6_2021 Alignment
# F_pk_right_image=[None] * 20
# F_pk_right_image[0]=0.5#-F_pk_right[19]
# for i in range(len(F_pk_right)-1):
#     F_pk_right_image[i+1]=-F_pk_right[i]

# f = plt.figure()    
# plt.bar(range(1,len(F_pk_right_image)+1),F_pk_right_image)
# plt.xlabel('N Division',**axis_font)
# plt.ylabel('$F_{p}$ [${\mu}m$]',**axis_font)
# plt.title('$F_{p}$ deviation, right flank',**axis_font)
# plt.grid()
# plt.xlim(0.5, len(F_pk_right_image)+0.5)
# plt.xticks(range(1, len(F_pk_right_image)+1))
# plt.ylim(-40,40)
# plt.tight_layout()
# os.chdir(gearwd)
# f.savefig("Fp for image.pdf", bbox_inches='tight')
# os.chdir(owd)
# plt.show()


#%%
"Control of lead profile"
# Import all the teeth
X_flank=[]
Y_flank=[]
Z_flank=[]

os.chdir(folder)
file = open('Cylinder 1 +0.000 mm.asc',"r") #It is cut by a cilinder, we get the points
os.chdir(owd)
for line in file:
    x, y, z = line.split()
    X_flank.append(float(x))  #need to assign float
    Y_flank.append(float(y))
    Z_flank.append(float(z))

fi_flank=[]
r_flank=[]
fi_flank_calc=0
#Divide the data by teeth
for i in range(len(X_flank)):
    fi_flank_calc=math.atan2(Y_flank[i],X_flank[i])*(180/np.pi)
    if fi_flank_calc<0:
        fi_flank_calc+=360
    fi_flank.append(fi_flank_calc)
    r_flank.append(np.sqrt(X_flank[i]**2+Y_flank[i]**2))


fi_flank_divided=[[] for x in range(N_teeth)]
r_flank_divided=[[] for x in range(N_teeth)]
X_flank_divided=[[] for x in range(N_teeth)]
Y_flank_divided=[[] for x in range(N_teeth)]
Z_flank_divided=[[] for x in range(N_teeth)]
for i in range(len(fi_flank)):
    if fi_flank[i]<fi_range/2 or fi_flank[i]>(360-(fi_range/2)):
        fi_flank_divided[0].append(fi_flank[i])
        r_flank_divided[0].append(r_flank[i])
        X_flank_divided[0].append(X_flank[i])
        Y_flank_divided[0].append(Y_flank[i])
        Z_flank_divided[0].append(Z_flank[i])
    else:
        for j in range(N_teeth):
            if fi_flank[i]>fi_range/2+(fi_range*(j-1)) and fi_flank[i]<(360-fi_range/2-fi_range*(N_teeth-1-j)):
                fi_flank_divided[j].append(fi_flank[i])
                r_flank_divided[j].append(r_flank[i])
                X_flank_divided[j].append(X_flank[i])
                Y_flank_divided[j].append(Y_flank[i])
                Z_flank_divided[j].append(Z_flank[i])
                                

for i in range(len(X_flank_divided)):           
    plt.plot(X_flank_divided[i],Y_flank_divided[i],marker='o', markerfacecolor='black', markersize=1, linestyle=' ')
plt.plot(0, 0,'ro',label='Origin C.S.')
plt.axis('equal')
plt.title('Gear_flank_divided')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()

#Only consider those in the evaluation length
#XXXXXXXXXXXXXXXXXX
"Val parameter"
val_length=0.9
#XXXXXXXXXXXXXXXXX
X_flank_val=[[] for x in range(N_teeth)]
Y_flank_val=[[] for x in range(N_teeth)]
Z_flank_val=[[] for x in range(N_teeth)]
fi_flank_val=[[] for x in range(N_teeth)]
r_flank_val=[[] for x in range(N_teeth)]
for i in range(N_teeth):
    for j in range(len(Z_flank_divided[i])):
        if Z_flank_divided[i][j]>np.min(Z_flank_divided[i])*val_length and Z_flank_divided[i][j]<np.max(Z_flank_divided[i])*val_length:
            X_flank_val[i].append(X_flank_divided[i][j])
            Y_flank_val[i].append(Y_flank_divided[i][j])
            Z_flank_val[i].append(Z_flank_divided[i][j])
            fi_flank_val[i].append(fi_flank_divided[i][j])
            r_flank_val[i].append(r_flank_divided[i][j])

        
#Now we divide them to right and left flanks
X_flank_left=[[] for x in range(N_teeth)]
Y_flank_left=[[] for x in range(N_teeth)]
Z_flank_left=[[] for x in range(N_teeth)]
X_flank_right=[[] for x in range(N_teeth)]
Y_flank_right=[[] for x in range(N_teeth)]
Z_flank_right=[[] for x in range(N_teeth)]
mid_ang=0       #needed for calculation
for i in range(1,N_teeth):
    mid_ang=(np.max(fi_flank_val[i])+np.min(fi_flank_val[i]))/2
    for j in range(len(fi_flank_val[i])):
        if fi_flank_val[i][j]>mid_ang:
            X_flank_left[i].append(X_flank_val[i][j])
            Y_flank_left[i].append(Y_flank_val[i][j])
            Z_flank_left[i].append(Z_flank_val[i][j])
        else:
            X_flank_right[i].append(X_flank_val[i][j])
            Y_flank_right[i].append(Y_flank_val[i][j])
            Z_flank_right[i].append(Z_flank_val[i][j])
            
for j in range(len(fi_flank_val[0])): #Here we also include the first tooth.
    if fi_flank_val[0][j]<(fi_range/2):
        X_flank_left[0].append(X_flank_val[0][j])
        Y_flank_left[0].append(Y_flank_val[0][j])
        Z_flank_left[0].append(Z_flank_val[0][j])
    else:
        X_flank_right[0].append(X_flank_val[0][j])
        Y_flank_right[0].append(Y_flank_val[0][j])
        Z_flank_right[0].append(Z_flank_val[0][j])
            

#for i in range(len(X_flank_left)):           
#    plt.plot(X_flank_left[i],Y_flank_left[i],marker='o', markerfacecolor='black', markersize=1, linestyle=' ')

# =============================================================================
# for i in range(len(X_flank_right)):           
#     plt.plot(X_flank_right[i],Y_flank_right[i],marker='o', markerfacecolor='black', markersize=1, linestyle=' ')
# plt.plot(0, 0,'ro',label='Origin C.S.')
# plt.axis('equal')
# plt.title('Gear_flank_divided_right')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid()
# plt.show()
# 
# =============================================================================
for i in range(len(X_flank_right)):           
    plt.plot(X_flank_right[i],Y_flank_right[i],marker='o', markerfacecolor='black', markersize=1, linestyle=' ')
plt.plot(0, 0,'ro',label='Origin C.S.')
plt.axis('equal')
plt.title('Gear_flank_divided_right')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()

# We are going to evaluate the differences in the X direction. They are also present in the Y direction.


#%%
'Calculations on right flanks'

"Circular coordinate system"
# First we transform it to the polar sistem
r_side_flank_right=[[] for x in range(N_teeth)]
fi_side_flank_right=[[] for x in range(N_teeth)]
angle=0
for i in range(N_teeth):
    for j in range(len(X_flank_right[i])):
        r_side_flank_right[i].append(math.sqrt(X_flank_right[i][j]**2 + Y_flank_right[i][j]**2))
        angle=math.atan2(Y_flank_right[i][j],X_flank_right[i][j])*(180/np.pi) #angle calculation, return 0<fi<360
        if angle<0:
            angle=angle+360
        fi_side_flank_right[i].append(angle)

# Then we nee to find the maximum angle on ach flank to set the base
index_min_side_right=0
index_min_side_right_list = []
X_base_flank_right=[]
Y_base_flank_right=[]
for i in range(N_teeth):
    index_min_side_right=np.argmin(fi_side_flank_right[i])  #maximal index of each flank
    index_min_side_right_list.append(index_min_side_right)  #a list of max indexes
    X_base_flank_right.append(X_flank_right[i][index_min_side_right])            #get the values
    Y_base_flank_right.append(Y_flank_right[i][index_min_side_right])


#Calculate the distances from the bases
distances_flank_right=[[] for x in range(N_teeth)]
for i in range(N_teeth):
    for j in range(len(X_flank_right[i])):
        distances_flank_right[i].append(np.sqrt((X_base_flank_right[i]-X_flank_right[i][j])**2+(Y_base_flank_right[i]-Y_flank_right[i][j])**2)*1000)


#XXXXXXXXXXXXX F_beta XXXXXXXXXXXXXX
F_beta_list_right=[]
for i in range(len(X_flank_right)):
    F_beta_list_right.append((np.max(distances_flank_right[i])-np.min(distances_flank_right[i])))

A_right=[]
k_right=[]
n_right=[]
for i in range(N_teeth):
    A=np.vstack([Z_flank_right[i], np.ones(len(Z_flank_right[i]))]).T
    A_right.append(A)   #Calculation of line parameters
    k, n = np.linalg.lstsq(A, distances_flank_right[i])[0]
    k_right.append(k)
    n_right.append(n)

curve_right=[[] for x in range(N_teeth)]
for i in range(N_teeth):
    for j in range(len(Z_flank_right[i])):
        curve_right[i].append(k_right[i]*Z_flank_right[i][j]+n_right[i])
        
distance=[[] for x in range(N_teeth)]
for i in range(N_teeth):
    for j in range(len(X_flank_right[i])):
        distance[i].append((n_right[i]+k_right[i]*Z_flank_right[i][j]-distances_flank_right[i][j])/(np.sqrt(1+k_right[i]*k_right[i])))

#curve up    
dist_ind_min=0
n_up=[]
for i in range(N_teeth):
    dist_ind_min=(np.argmin(distance[i]))
    n_up.append(distances_flank_right[i][dist_ind_min]-k_right[i]*Z_flank_right[i][dist_ind_min])
curve_up_right=[[] for x in range(N_teeth)]
for i in range(N_teeth):
    for j in range(len(Z_flank_right[i])):
        curve_up_right[i].append(k_right[i]*Z_flank_right[i][j]+n_up[i])
#curve down        
dist_ind_max=0
n_down=[]
for i in range(N_teeth):
    dist_ind_max=(np.argmax(distance[i]))
    n_down.append(distances_flank_right[i][dist_ind_max]-k_right[i]*Z_flank_right[i][dist_ind_max])
curve_down_right=[[] for x in range(N_teeth)]
for i in range(N_teeth):
    for j in range(len(Z_flank_right[i])):
        curve_down_right[i].append(k_right[i]*Z_flank_right[i][j]+n_down[i])

#XXXXXXXXXXXXX f_f_beta XXXXXXXXXXXXXX
f_f_beta_right=[]
for i in range(N_teeth):    
    f_f_beta_right.append((n_up[i]-n_down[i]))

#XXXXXXXXXXXXX f_H_beta XXXXXXXXXXXXXX
f_H_beta_right=[]
for i in range(N_teeth):
    if k_right[i]>=0:    
        f_H_beta_right.append((np.max(curve_right[i])-np.min(curve_right[i])))
    else:
        f_H_beta_right.append(-(np.max(curve_right[i])-np.min(curve_right[i])))
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


'Calculations on left flanks'
"Circular coordinate system"
# First we transform it to the polar sistem
r_side_flank_left=[[] for x in range(N_teeth)]
fi_side_flank_left=[[] for x in range(N_teeth)]
angle=0
for i in range(N_teeth):
    for j in range(len(X_flank_left[i])):
        r_side_flank_left[i].append(math.sqrt(X_flank_left[i][j]**2 + Y_flank_left[i][j]**2))
        angle=math.atan2(Y_flank_left[i][j],X_flank_left[i][j])*(180/np.pi) #angle calculation, return 0<fi<360
        if angle<0:
            angle=angle+360
        fi_side_flank_left[i].append(angle)

# Then we nee to find the maksimum angle on ach flank to set the base
index_max_side_left=0
index_max_side_left_list = []
X_base_flank_left=[]
Y_base_flank_left=[]
for i in range(N_teeth):
    index_max_side_left=np.argmax(fi_side_flank_left[i])  #maximal index of each flank
    index_max_side_left_list.append(index_max_side_left)  #a list of max indexes
    X_base_flank_left.append(X_flank_left[i][index_max_side_left])            #get the values
    Y_base_flank_left.append(Y_flank_left[i][index_max_side_left])


#Calculate the distances from the bases
distances_flank_left=[[] for x in range(N_teeth)]
for i in range(N_teeth):
    for j in range(len(X_flank_left[i])):
        distances_flank_left[i].append(np.sqrt((X_base_flank_left[i]-X_flank_left[i][j])**2+(Y_base_flank_left[i]-Y_flank_left[i][j])**2)*1000)


#plt.plot(distances_flank_left[0],Z_flank_left[0])
#maksimum_points_flank_left=[]
#for i in range(N_teeth):
#    for j in range(len(X_flank_left[i])):
#        distance_flank_list_left[i].append(((X_flank_left[i][j])**2+(Y_flank_left[i][j])**2)**(1/2))

#XXXXXXXXXXXXX F_beta XXXXXXXXXXXXXX
F_beta_list_left=[]
for i in range(len(X_flank_left)):
    F_beta_list_left.append((np.max(distances_flank_left[i])-np.min(distances_flank_left[i])))

A_left=[]
k_left=[]
n_left=[]
for i in range(N_teeth):
    A=np.vstack([Z_flank_left[i], np.ones(len(Z_flank_left[i]))]).T
    A_left.append(A)   #Calculation of line parameters
    k, n = np.linalg.lstsq(A, distances_flank_left[i])[0]
    k_left.append(k)
    n_left.append(n)

curve_left=[[] for x in range(N_teeth)]
for i in range(N_teeth):
    for j in range(len(Z_flank_left[i])):
        curve_left[i].append(k_left[i]*Z_flank_left[i][j]+n_left[i])
        
distance=[[] for x in range(N_teeth)]
for i in range(N_teeth):
    for j in range(len(X_flank_left[i])):
        distance[i].append((n_left[i]+k_left[i]*Z_flank_left[i][j]-distances_flank_left[i][j])/(np.sqrt(1+k_left[i]*k_left[i])))

#curve_left up    
dist_ind_min=0
n_up=[]
for i in range(N_teeth):
    dist_ind_min=(np.argmin(distance[i]))
    n_up.append(distances_flank_left[i][dist_ind_min]-k_left[i]*Z_flank_left[i][dist_ind_min])
curve_up_left=[[] for x in range(N_teeth)]
for i in range(N_teeth):
    for j in range(len(Z_flank_left[i])):
        curve_up_left[i].append(k_left[i]*Z_flank_left[i][j]+n_up[i])
#curve_left down        
dist_ind_max=0
n_down=[]
for i in range(N_teeth):
    dist_ind_max=(np.argmax(distance[i]))
    n_down.append(distances_flank_left[i][dist_ind_max]-k_left[i]*Z_flank_left[i][dist_ind_max])
curve_down_left=[[] for x in range(N_teeth)]
for i in range(N_teeth):
    for j in range(len(Z_flank_left[i])):
        curve_down_left[i].append(k_left[i]*Z_flank_left[i][j]+n_down[i])

#XXXXXXXXXXXXX f_f_beta XXXXXXXXXXXXXX
f_f_beta_left=[]
for i in range(N_teeth):    
    f_f_beta_left.append((n_up[i]-n_down[i]))

#XXXXXXXXXXXXX f_H_beta XXXXXXXXXXXXXX
f_H_beta_left=[]
for i in range(N_teeth):
    if k_left[i]>=0:    
        f_H_beta_left.append((np.max(curve_left[i])-np.min(curve_left[i])))
    else:
        f_H_beta_left.append(-(np.max(curve_left[i])-np.min(curve_left[i])))

#%%
"Graphs and table generation"
#F_beta
test=19
side="right"
sideX=distances_flank_right
sideZ=Z_flank_right
curve=curve_right
curve_up=curve_up_right
curve_down=curve_down_right

axis_font = {'fontname':'Times new Roman', 'size':'16'}

plt.plot(sideZ[test],sideX[test],marker='o', markersize=1, linestyle=' ')
plt.axhline(y=np.max(sideX[test]), color='k', linestyle='--',linewidth=1)
plt.axhline(y=np.min(sideX[test]), color='k', linestyle='--',linewidth=1)
string="Lead profile deviation on tooth %d, %s side, $F_{beta}$" %(test,side)
plt.title(string,**axis_font)
plt.xlabel('Z [mm]',**axis_font)
plt.ylabel('distance [$\mu$m]',**axis_font)
plt.grid()
plt.tight_layout()
os.chdir(gearwd)
plt.savefig("F_beta.png", dpi = 200)
os.chdir(owd)
plt.show()
F_beta=(np.max(sideX[test])-np.min(sideX[test]))
print('F_beta=',round(F_beta,1),'um')

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
f = plt.figure()
plt.plot(sideZ[test],sideX[test],marker='o', markersize=1, linestyle=' ',label="Lead profile deviation")
plt.plot(sideZ[test],curve[test],'k', linestyle='--',linewidth=0.5,label="Least square line")
plt.plot(sideZ[test],curve_up[test],'r', linestyle='--', linewidth=0.5,label="Min. and max. lines")
plt.plot(sideZ[test],curve_down[test],'r', linestyle='--',linewidth=0.5)
string="Lead profile deviation on tooth %d, %s side, $f_{f beta}$" %(test,side)
plt.title(string,**axis_font)
plt.xlabel('Z [mm]',**axis_font)
plt.ylabel('distance [$\mu$m]',**axis_font)
font = font_manager.FontProperties(family='Times New Roman',
                                   style='normal', size=16)

plt.legend(prop=font)
plt.grid()
plt.tight_layout()
os.chdir(gearwd)
plt.savefig("f_f_beta.png", dpi = 200)
os.chdir(owd)
plt.show()
f.savefig("Lead profile deviation on tooth.pdf", bbox_inches='tight')
f_f_beta=(n_up[test]-n_down[test])
print('f_f_beta=',round(f_f_beta,1),'um')


#==============================================================================
#F_H_beta
plt.plot(sideZ[test],sideX[test],marker='o', markersize=1, linestyle=' ')
plt.plot(sideZ[test],curve[test],'k', linestyle='--',linewidth=0.5)
plt.axhline(y=np.min(curve[test]), color='k', linestyle='--',linewidth=1)
plt.axhline(y=np.max(curve[test]), color='k', linestyle='--',linewidth=1)
string="Lead profile deviation on tooth %d, %s side, $f_{H {beta}}$" %(test,side)
plt.title(string,**axis_font)
plt.xlabel('Z [mm]',**axis_font)
plt.ylabel('distance [$\mu$m]',**axis_font)
plt.grid()
plt.tight_layout()
os.chdir(gearwd)
plt.savefig("f_H_beta.png", dpi = 200)
os.chdir(owd)
plt.show()
f_H_beta=(np.max(curve[test])-np.min(curve[test]))
print('f_H_beta=',round(f_H_beta,1),'um')

#==============================================================================    
"Generate a table"
t = PrettyTable()
t.field_names = ['Tooth', '$F_beta_right$','$F_beta_left$','$f_{f_beta_right}$','$f_{f_beta_left}$','$f_{H_beta_right}$','$f_{H_beta_left}$']
for i in range(N_teeth):
    t.add_row([i, round(F_beta_list_right[i],1),round(F_beta_list_left[i],1),
               round(f_f_beta_right[i],1),round(f_f_beta_left[i],1),
               round(f_H_beta_right[i],1),round(f_H_beta_left[i],1)])
print (t)
#==============================================================================

circle_i=np.arange(0,np.pi/6,0.01)
x_for_circle_i=r_pc*np.cos(circle_i)
y_for_circle_i=r_pc*np.sin(circle_i)
plt.plot(x_for_circle_i,y_for_circle_i, color='orange', linestyle='-.',label="Pitch circle")
#plt.plot([X_divided_val[5][135],X_divided_val[5][136]],[Y_divided_val[5][135],Y_divided_val[5][136]],linestyle='-')

#for i in range(0,85):  
plt.plot(X_divided_val[0],Y_divided_val[0], color='black', linestyle='-', label="Tooth middle profile ")#marker='o', markersize=1) #, marker='o', markersize=1)

plt.plot(X_flank_left[0],Y_flank_left[0], marker='o', color='red', markersize=2,linestyle=' ', label="Lead profile points along Z")
plt.axis('equal')
#plt.title('Lead profile control',**axis_font)
plt.xlabel('$x$ [mm]',**axis_font)
plt.ylabel('$y$ [mm]',**axis_font)
plt.legend(prop={'family':'Times new Roman', 'size':16})
plt.grid()
plt.tight_layout()
axes = plt.gca()
axes.set_xlim([9,11.5])
axes.set_ylim([0.25,1.5])
os.chdir(gearwd)
plt.savefig("lead_profile_control.png", dpi = 200)
os.chdir(owd)
plt.show()


# Plot showing the points on one tooth
plt.plot(x_for_circle_i,y_for_circle_i, color='orange', linestyle='-.',label="Pitch circle")
#plt.plot([X_divided_val[5][135],X_divided_val[5][136]],[Y_divided_val[5][135],Y_divided_val[5][136]],linestyle='-')
#for i in range(0,85):  
plt.plot(X_divided_val[0],Y_divided_val[0], color='black', linestyle='-', label="Tooth middle profile ")#marker='o', markersize=1) #, marker='o', markersize=1)
plt.plot(X_flank_left[0],Y_flank_left[0], marker='o', color='red', markersize=2,linestyle=' ', label="Lead profile points along Z")
plt.axis('equal')
#plt.title('Lead profile control',**axis_font)
plt.xlabel('$x$ [mm]',**axis_font)
plt.ylabel('$y$ [mm]',**axis_font)
#plt.legend(prop={'family':'Times new Roman', 'size':16})
plt.grid()
plt.tight_layout()
axes = plt.gca()
axes.set_xlim([9.97,9.97])
axes.set_ylim([0.76,0.795])
os.chdir(gearwd)
plt.savefig("lead_profile_control_closeup.png", dpi = 200)
os.chdir(owd)
plt.show()
#%%
#==============================================================================
#INVOLUTE EVALUATION
#==============================================================================
# Evaluation of the involute. First we determine the evaluation length
radius_b=(module*N_teeth*np.cos(np.radians(alpha)))/2

val_length_cutoff=0.04  
up_limit=0
down_limit=0
X_divided_cutoff=[[] for x in range(N_teeth)]
Y_divided_cutoff=[[] for x in range(N_teeth)]
r_divided_cutoff=[[] for x in range(N_teeth)]
fi_divided_cutoff=[[] for x in range(N_teeth)]
for i in range(len(X_divided_val)):
    down_limit=radius_b+(np.max(r_divided[i])-radius_b)*val_length_cutoff*2       #define the radius for the cutoff
    up_limit=np.max(r_divided[i])-(np.max(r_divided[i])-radius_b)*val_length_cutoff*2
    # up_limit=np.max(r_divided[i])-(np.max(r_divided[i])-np.min(r_divided[i]))*val_length_cutoff #here we set the limit by radius
    # down_limit=np.min(r_divided[i])+(np.max(r_divided[i])-np.min(r_divided[i]))*val_length_cutoff*6   #temporary, normally this is 1
    for j in range(len(X_divided_val[i])):
        r_divid_val=np.sqrt(X_divided_val[i][j]**2+Y_divided_val[i][j]**2)
        if r_divid_val<=up_limit and r_divid_val>=down_limit: #and r_divid_val>radius_b:          # Temporary only consideres values bigger than radius_b
            X_divided_cutoff[i].append(X_divided_val[i][j])
            Y_divided_cutoff[i].append(Y_divided_val[i][j])
            r_divided_cutoff[i].append(np.sqrt(X_divided_val[i][j]**2+Y_divided_val[i][j]**2))
            fi_divided_cutoff[i].append(math.atan2(Y_divided_val[i][j],X_divided_val[i][j])*(180/np.pi))
for i in range(len(X_divided_cutoff)):           
    plt.plot(X_divided_cutoff[i],Y_divided_cutoff[i],marker='o', markerfacecolor='black', markersize=1, linestyle=' ')
plt.plot(0, 0,'ro',label='Origin C.S.')
plt.axis('equal')
plt.title('Divided teeth, cutoff')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()

#%%
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

for i in range(len(fi_divided_cutoff)):
    for j in range(len(fi_divided_cutoff[i])):
        if fi_divided_cutoff[i][j]<0:
            fi_divided_cutoff[i][j]=fi_divided_cutoff[i][j]+360 #add 360 to get values from 0 to 360

X_divided_cutoff_new=[[] for x in range(N_teeth)] #!!!!!! Prepare new X and Y for 
Y_divided_cutoff_new=[[] for x in range(N_teeth)] #!!!!!!
for i in range(len(fi_divided_cutoff)):
    for j in range(len(fi_divided_cutoff[i])):
        X_divided_cutoff_new[i].append(r_divided_cutoff[i][j]*np.cos(np.radians(fi_divided_cutoff[i][j])))
        Y_divided_cutoff_new[i].append(r_divided_cutoff[i][j]*np.sin(np.radians(fi_divided_cutoff[i][j])))

# Plot the new values to check
# =============================================================================
# for i in range(len(X_divided_cutoff)): 
#     plt.plot(X_divided_cutoff_new[i],Y_divided_cutoff_new[i],marker='o', markerfacecolor='black', markersize=1, linestyle=' ')
# plt.plot(0, 0,'ro',label='Origin C.S.')
# plt.axis('equal')
# plt.title('Divided teeth, cutoff, new')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid()
# plt.show()
# =============================================================================

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#Lets divide to left and right
r_cutoff_right=[[] for x in range(N_teeth)]
fi_cutoff_right=[[] for x in range(N_teeth)]
X_cutoff_right=[[] for x in range(N_teeth)]
Y_cutoff_right=[[] for x in range(N_teeth)]
r_cutoff_left=[[] for x in range(N_teeth)]
fi_cutoff_left=[[] for x in range(N_teeth)]
X_cutoff_left=[[] for x in range(N_teeth)]
Y_cutoff_left=[[] for x in range(N_teeth)]
avg_angle=0

for j in range(len(r_divided_cutoff[0])): #this is the first tooth
    if fi_divided_cutoff[0][j]>180:
        r_cutoff_right[0].append(r_divided_cutoff[0][j])
        fi_cutoff_right[0].append(fi_divided_cutoff[0][j])
        X_cutoff_right[0].append(X_divided_cutoff_new[0][j])
        Y_cutoff_right[0].append(Y_divided_cutoff_new[0][j])
    else:
        r_cutoff_left[0].append(r_divided_cutoff[0][j])
        fi_cutoff_left[0].append(fi_divided_cutoff[0][j])
        X_cutoff_left[0].append(X_divided_cutoff_new[0][j])
        Y_cutoff_left[0].append(Y_divided_cutoff_new[0][j])
            
for i in range(1,N_teeth): #the remaining teeth
    avg_angle=np.average(fi_divided_cutoff[i])
    for j in range(len(r_divided_cutoff[i])):
        if fi_divided_cutoff[i][j]<avg_angle:
            r_cutoff_right[i].append(r_divided_cutoff[i][j])
            fi_cutoff_right[i].append(fi_divided_cutoff[i][j])
            X_cutoff_right[i].append(X_divided_cutoff_new[i][j])
            Y_cutoff_right[i].append(Y_divided_cutoff_new[i][j])
        else:
            r_cutoff_left[i].append(r_divided_cutoff[i][j])
            fi_cutoff_left[i].append(fi_divided_cutoff[i][j])
            X_cutoff_left[i].append(X_divided_cutoff_new[i][j])
            Y_cutoff_left[i].append(Y_divided_cutoff_new[i][j])

# =============================================================================
# for i in range(len(X_cutoff_right)):           
#     plt.plot(X_cutoff_right[i],Y_cutoff_right[i],marker='o', markerfacecolor='black', markersize=1, linestyle=' ')
# plt.plot(0, 0,'ro',label='Origin C.S.')
# plt.axis('equal')
# plt.title('Divided teeth, cutoff, right')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid()
# plt.show()
# 
# for i in range(len(X_cutoff_left)):           #
#     plt.plot(X_cutoff_left[i],Y_cutoff_left[i],marker='o', markerfacecolor='black', markersize=1, linestyle=' ')
# plt.plot(0, 0,'ro',label='Origin C.S.')
# plt.axis('equal')
# plt.title('Divided teeth, cutoff, left')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid()
# plt.show()
# =============================================================================
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

#==============================================================================
#Now we start with the calculation of the parameters
# =============================================================================
# For the right flank we need to rotate the teeth over 180 to the first and second quadrant. Our method works there.
# Lets also mirror the left flank to the right, so that we can evaluate it. It works for the right flank.
index_min=0
fi_cut_calc=0
r_cutoff_left_mirr=[[] for x in range(N_teeth)]
fi_cutoff_left_mirr=[[] for x in range(N_teeth)]
X_cutoff_left_mirr=[[] for x in range(N_teeth)]
Y_cutoff_left_mirr=[[] for x in range(N_teeth)]
for i in range(len(X_cutoff_left)): 
    index_min = np.argmin(r_cutoff_left[i])
    for j in range(len(r_cutoff_left[i])):
        r_cutoff_left_mirr[i].append(r_cutoff_left[i][j]) #mirror and also Y>0
        fi_cut_calc=fi_cutoff_left[i][index_min]+(fi_cutoff_left[i][index_min]-fi_cutoff_left[i][j])
        if fi_cut_calc>180:
            fi_cut_calc=fi_cut_calc-180
        else:
            fi_cut_calc=fi_cut_calc
        if fi_cut_calc>90:
            fi_cut_calc=fi_cut_calc-90
        else:
            fi_cut_calc=fi_cut_calc
        fi_cutoff_left_mirr[i].append(fi_cut_calc)
        X_cutoff_left_mirr[i].append(r_cutoff_left[i][j]*np.cos(np.radians(fi_cut_calc)))
        Y_cutoff_left_mirr[i].append(r_cutoff_left[i][j]*np.sin(np.radians(fi_cut_calc)))

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Checking how it rotated
for i in range(0,int(len(X_cutoff_left)/4)):           
    plt.plot(X_cutoff_left[i],Y_cutoff_left[i],marker='o', markerfacecolor='black', markersize=1, linestyle=' ')
for i in range(len(X_cutoff_left_mirr)):           
    plt.plot(X_cutoff_left_mirr[i],Y_cutoff_left_mirr[i],marker='o', markerfacecolor='black', markersize=1, linestyle=' ')
plt.plot(0, 0,'ro',label='Origin C.S.')
plt.axis('equal')
plt.title('Divided teeth, cutoff, left+mirrored')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

#rotate the right flank so its Y>0
fi_cut_calc=0
r_cutoff_right_mirr=[[] for x in range(N_teeth)]
fi_cutoff_right_mirr=[[] for x in range(N_teeth)]
X_cutoff_right_mirr=[[] for x in range(N_teeth)]
Y_cutoff_right_mirr=[[] for x in range(N_teeth)]
for i in range(len(fi_cutoff_right)):
    for j in range(len(fi_cutoff_right[i])):
        if fi_cutoff_right[i][j]>180:
            fi_cut_calc=fi_cutoff_right[i][j]-180
        else:
            fi_cut_calc=fi_cutoff_right[i][j]
        if fi_cut_calc>88:                          #this is 90 by default
            fi_cut_calc=fi_cut_calc-90
        else:
            fi_cut_calc=fi_cut_calc
        r_cutoff_right_mirr[i].append(r_cutoff_right[i][j])
        fi_cutoff_right_mirr[i].append(fi_cut_calc)
        X_cutoff_right_mirr[i].append(r_cutoff_right[i][j]*np.cos(np.radians(fi_cut_calc)))
        Y_cutoff_right_mirr[i].append(r_cutoff_right[i][j]*np.sin(np.radians(fi_cut_calc)))

#the right flank of tooth 0 is now rotated to -x, only use for calculation
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx        
for i in range(0,len(X_cutoff_left)):           
    plt.plot(X_cutoff_right_mirr[i],Y_cutoff_right_mirr[i],marker='o', markerfacecolor='black', markersize=1, linestyle=' ')
plt.plot(0, 0,'ro',label='Origin C.S.')
plt.axis('equal')
plt.title('Divided teeth, cutoff, right rotated')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()  
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
razdalja=[[] for x in range(N_teeth)]    #temporary testiranje
# RIGHT FLANK
y_T_right=[[] for x in range(N_teeth)]
x_T_right=[[] for x in range(N_teeth)]
v_T_right=0
A_x_right=[[] for x in range(N_teeth)]
A_y_right=[[] for x in range(N_teeth)]
x_plot_right=[[] for x in range(N_teeth)]
dist_MA_right=[[] for x in range(N_teeth)]
for i in range(len(X_cutoff_right_mirr)):
    for j in range(len(X_cutoff_right_mirr[i])):
        y__T=(((radius_b**2)*Y_cutoff_right_mirr[i][j]+radius_b*X_cutoff_right_mirr[i][j]*np.sqrt(X_cutoff_right_mirr[i][j]**2+Y_cutoff_right_mirr[i][j]**2-(radius_b**2)))/(X_cutoff_right_mirr[i][j]**2+Y_cutoff_right_mirr[i][j]**2))
        x__T=((radius_b**2)-Y_cutoff_right_mirr[i][j]*y__T)/X_cutoff_right_mirr[i][j]
        v_T=np.arccos(x__T/radius_b)
# =============================================================================
#         if Y_cutoff_right_mirr[i][j]>0:
#             v_T=np.arccos(x__T/radius_b)
#         else:
#             v_T=np.arccos(x__T/radius_b)
# =============================================================================
        A_x_right[i].append(radius_b*(np.cos(v_T)+v_T*np.sin(v_T)))
        A_y_right[i].append(radius_b*(np.sin(v_T)-v_T*np.cos(v_T)))
        dist_MA_right[i].append(np.sqrt((x__T-X_cutoff_right_mirr[i][j])**2+(y__T-Y_cutoff_right_mirr[i][j])**2)-v_T*radius_b)
        #testna_dolzina[i].append(np.sqrt((x__T-radius_b*(np.cos(v_T)+v_T*np.sin(v_T)))**2+(y__T-radius_b*(np.sin(v_T)-v_T*np.cos(v_T)))**2))    #temporary testiranje
        x_T_right[i].append(x__T)
        y_T_right[i].append(y__T)
        x_plot_right[i].append(v_T*radius_b)
        
# #%% #temporary to je za testirat roll length bilo

# plt.plot(X_cutoff_right[10],Y_cutoff_right[10],marker='o', markersize=4, linestyle=' ')
# plt.axis('equal')
# plt.title('Determining profile deviation',**axis_font)
# plt.xlabel('$x$ [mm]',**axis_font)
# plt.ylabel('$y$ [mm]',**axis_font)
# plt.grid()
# plt.show() 
     
# #%%

# # temporary to je za testirat roll length bilo
# x_plot_right_nov=[[] for x in range(N_teeth)]
# for i in range(len(X_cutoff_right_mirr)):
#     for j in range(len(X_cutoff_right_mirr[i])):
#         x_plot_right_nov[i].append(x_plot_right[i][j]/3)
# A_x=radius_b*(np.cos(v_T)+v_T*np.sin(v_T))
#         A_y=radius_b*(np.sin(v_T)-v_T*np.cos(v_T))
#         razdalja[i].append(np.sqrt((A_x-x__T)**2+(A_y-y__T)**2))
# x_plot_right=razdalja

# =============================================================================
# LEFT FLANK
y_T_left=[[] for x in range(N_teeth)]
x_T_left=[[] for x in range(N_teeth)]
v_T_left=0
A_x_left=[[] for x in range(N_teeth)]
A_y_left=[[] for x in range(N_teeth)]
x_plot_left=[[] for x in range(N_teeth)]
dist_MA_left=[[] for x in range(N_teeth)]
for i in range(len(X_cutoff_left_mirr)):
    for j in range(len(X_cutoff_left_mirr[i])):
        y__T=(((radius_b**2)*Y_cutoff_left_mirr[i][j]+radius_b*X_cutoff_left_mirr[i][j]*np.sqrt(X_cutoff_left_mirr[i][j]**2+Y_cutoff_left_mirr[i][j]**2-(radius_b**2)))/(X_cutoff_left_mirr[i][j]**2+Y_cutoff_left_mirr[i][j]**2))
        x__T=((radius_b**2)-Y_cutoff_left_mirr[i][j]*y__T)/X_cutoff_left_mirr[i][j]
        v_T=np.arccos(x__T/radius_b)
# =============================================================================
#         if Y_cutoff_left[i][j]>0:
#             v_T=np.arccos(x__T/radius_b)
#         else:
#             v_T=np.arccos(x__T/radius_b)
# =============================================================================
        A_x_left[i].append(radius_b*(np.cos(v_T)+v_T*np.sin(v_T)))
        A_y_left[i].append(radius_b*(np.sin(v_T)-v_T*np.cos(v_T)))
        dist_MA_left[i].append(np.sqrt((x__T-X_cutoff_left_mirr[i][j])**2+(y__T-Y_cutoff_left_mirr[i][j])**2)-v_T*radius_b)
        x_T_left[i].append(x__T)
        y_T_left[i].append(y__T)
        x_plot_left[i].append(v_T*radius_b)

# =============================================================================

min_x_plot_right=0
for i in range(len(x_plot_right)):
    min_x_plot_right=np.min(x_plot_right[i])
    for j in range(len(x_plot_right[i])):
        x_plot_right[i][j]=x_plot_right[i][j]-min_x_plot_right
        
min_x_plot_left=0
for i in range(len(x_plot_left)):
    min_x_plot_left=np.min(x_plot_left[i])
    for j in range(len(x_plot_left[i])):
        x_plot_left[i][j]=x_plot_left[i][j]-min_x_plot_left

#%%
#For LATEX
# =============================================================================
# axis_font = {'fontname':'Times new Roman', 'size':'25'}
#         
# for i in range(len(dist_MA_right[3])):
#     #some_value_dist_MA=(np.max(dist_MA_right[3])+np.min(dist_MA_right[3]))/2
#     #dist_MA_right[4][i]=dist_MA_right[4][i]-some_value_dist_MA
#     dist_MA_right[3][i]=(dist_MA_right[3][i]-np.min(dist_MA_right[3]))*1000 +3.4
# =============================================================================
#%%
"Results & graphs for the selected involute"
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Determine what graphs to display
test=10
side="right"
sideX=X_cutoff_right_mirr
sideY=Y_cutoff_right_mirr
sideAx=A_x_right
sideAy=A_y_right
sideTx=x_T_right
sideTy=y_T_right
sideMA=dist_MA_right
sideX_plot=x_plot_right
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

plt.plot(sideX[test],sideY[test],marker='o',color='red', markersize=2, linestyle=' ',label='Measured values M')
for_circ=np.arange(0,np.pi/3,0.1)
x_for_circ=radius_b*np.cos(for_circ)
y_for_circ=radius_b*np.sin(for_circ)    
plt.plot(0, 0, marker='x',color='red',label='Origin', markersize=2,linestyle=' ')
plt.plot(x_for_circ,y_for_circ, color='orange', linestyle='-.', label='Base circle')
plt.plot(sideAx[test],sideAy[test],marker='o',color='blue', markersize=2, linestyle=' ',label='Theoretical Involute A')
plt.plot(sideTx[test],sideTy[test],marker='s',color='black', markersize=2, linestyle=' ',label='Values on base circle T')
plt.axis('equal')
plt.title('Determining profile deviation',**axis_font)
plt.xlabel('$x$ [mm]',**axis_font)
plt.ylabel('$y$ [mm]',**axis_font)
plt.grid()
plt.legend(prop={'family':'Times new Roman', 'size':16},markerscale=3)
plt.tight_layout()
os.chdir(gearwd)
plt.savefig("Profile control.png", dpi = 200)
os.chdir(owd)
plt.show()

plt.plot(sideMA[test],marker='o', markerfacecolor='black', markersize=1, linestyle='-',label='distance_1')
#plt.plot(d_lot,marker='o', markerfacecolor='black', markersize=1, linestyle='-',label='d_lot')
plt.title('Distances')
plt.xlabel('count')
plt.ylabel('distance')
plt.grid()
plt.legend()
plt.show()

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"Graphs"
#F_alpha

#for the plot to micrometers
for i in range(len(sideMA[test])):
    sideMA[test][i]=(sideMA[test][i]-np.min(sideMA[test]))*1000  #-np.min(...) je temporary) *1000 je vredu?

plt.plot(sideX_plot[test],sideMA[test],marker='o', markersize=4, linestyle=' ')
plt.axhline(y=np.max(sideMA[test]), color='k', linestyle='--',linewidth=1)
plt.axhline(y=np.min(sideMA[test]), color='k', linestyle='--',linewidth=1)

plt.text(0, np.max(sideMA[test]), round(np.max(sideMA[test]),1),**axis_font)
plt.text(0, np.min(sideMA[test]), round(np.min(sideMA[test]),1),**axis_font)

string="Involute on tooth %d, %s, $F_{alpha}$" %(test,side)
#string="Involute on tooth 4, %s, $F_{alpha}$" %(side)
plt.title(string,**axis_font)
plt.xlabel('Profile roll length [mm]',**axis_font)
#plt.ylabel('Dist. to theoretical profile [um]',**axis_font)
plt.ylabel('Deviation [um]',**axis_font)
plt.grid()
plt.tight_layout()
os.chdir(gearwd)
plt.savefig("F_alpha.png", dpi = 300)
axes = plt.gca()
axes.set_ylim([0,50])
os.chdir(owd)
plt.show()

#reset
for i in range(len(sideMA[test])):
    sideMA[test][i]=(sideMA[test][i])/1000

F_alpha=(np.max(sideMA[test])-np.min(sideMA[test]))*1000
print('F_alpha=',round(F_alpha,1),'um')

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
A = np.vstack([sideX_plot[test], np.ones(len(sideX_plot[test]))]).T   #Calculation of line parameters
k, n = np.linalg.lstsq(A, sideMA[test])[0]

curve=[]
for i in range(len(sideX_plot[test])):
     curve.append(k*sideX_plot[test][i]+n)

plt.plot(sideX_plot[test],sideMA[test],marker='o', markersize=1, linestyle=' ')
plt.plot(sideX_plot[test],curve, linestyle='--')
string="Involute, %s, tooth %d, $F_{alpha}$" %(side,test)
plt.title(string)
plt.xlabel('Profile roll length [mm]')
plt.ylabel('distance')
plt.grid()
plt.show()

#==============================================================================
distance=[]
for i in range(len(sideMA[test])):
    distance.append((n+k*sideX_plot[test][i]-sideMA[test][i])/(np.sqrt(1+k*k)))

#==============================================================================
#==============================================================================
dist_ind_min=np.argmin(distance)
n_up=sideMA[test][dist_ind_min]-k*sideX_plot[test][dist_ind_min]
curve_up=[]
for i in range(len(sideX_plot[test])):
     curve_up.append(k*sideX_plot[test][i]+n_up)

dist_ind_max=np.argmax(distance)
n_down=sideMA[test][dist_ind_max]-k*sideX_plot[test][dist_ind_max]
curve_down=[]
for i in range(len(sideX_plot[test])):
     curve_down.append(k*sideX_plot[test][i]+n_down)

plt.plot(sideX_plot[test],sideMA[test],marker='o', markersize=1, linestyle=' ')
plt.plot(sideX_plot[test],curve,'k', linestyle='--',linewidth=0.5)
plt.plot(sideX_plot[test],curve_up,'r', linestyle='--', linewidth=0.5)
plt.plot(sideX_plot[test],curve_down,'r', linestyle='--',linewidth=0.5)
string="Involute, %s, tooth %d, $f_{f alpha}$" %(side,test)
plt.title(string)
plt.xticks(rotation=90)
plt.xlabel('Profile roll length [mm]')
plt.ylabel('Distance from theoretical involute [mm]')
plt.grid()
plt.tight_layout()
os.chdir(gearwd)
plt.savefig("f_f_alpha.png", dpi = 150)
os.chdir(owd)
plt.show()
f_f_alpha=(n_up-n_down)*1000
print('f_f_alpha=',round(f_f_alpha,1),'um')


#==============================================================================
#F_H_alpha
"""
for i in range(len(sideMA[test])):
    sideMA[test][i]=sideMA[test][i]+3.84
    curve[i]=curve[i]+3.84
"""
#For the report plot
f = plt.figure()
#==============================================================================
for i in range(len(sideMA[test])):
    sideMA[test][i]=(sideMA[test][i])*1000

A = np.vstack([sideX_plot[test], np.ones(len(sideX_plot[test]))]).T   #Calculation of line parameters
k, n = np.linalg.lstsq(A, sideMA[test])[0]

curve=[]
for i in range(len(sideX_plot[test])):
     curve.append(k*sideX_plot[test][i]+n)

distance=[]
for i in range(len(sideMA[test])):
    distance.append((n+k*sideX_plot[test][i]-sideMA[test][i])/(np.sqrt(1+k*k)))

#==============================================================================

dist_ind_min=np.argmin(distance)
n_up=sideMA[test][dist_ind_min]-k*sideX_plot[test][dist_ind_min]
curve_up=[]
for i in range(len(sideX_plot[test])):
     curve_up.append(k*sideX_plot[test][i]+n_up)

dist_ind_max=np.argmax(distance)
n_down=sideMA[test][dist_ind_max]-k*sideX_plot[test][dist_ind_max]
curve_down=[]
for i in range(len(sideX_plot[test])):
     curve_down.append(k*sideX_plot[test][i]+n_down)
plt.plot(sideX_plot[test],curve_up,'r', linestyle='--', linewidth=1)
plt.plot(sideX_plot[test],curve_down,'r', linestyle='--',linewidth=1)
#For the report plot
#==============================================================================
 
csfont = {'fontname':'Times New Roman'}
plt.plot(sideX_plot[test],sideMA[test],marker='o', markersize=3, color='k', linestyle=' ', label='profile deviation')
plt.plot(sideX_plot[test],curve,'k', linestyle='-.',linewidth=1.5, label='least square fit')
plt.axhline(y=np.min(curve), color='k', linestyle='--',linewidth=1)
plt.axhline(y=np.max(curve), color='k', linestyle='--',linewidth=1)
#string="Left profile, tooth 14, $f_{H alpha}$"
#string="Involute, %s, tooth %d, $f_{H alpha}$" %(side,test)
#plt.title(string,fontsize=15,**csfont)
plt.title('Determining profile deviations',fontsize=15,**csfont)
plt.xlabel('Profile roll length [mm]',fontsize=15,**csfont)
plt.ylabel('Dist. to theo. profile [µm]',fontsize=15,**csfont)
plt.grid()
font = font_manager.FontProperties(family='Times New Roman',
                                   style='normal', size=15)

plt.legend(prop=font)
plt.tight_layout()
os.chdir(gearwd)
plt.savefig("f_H_alpha.png", dpi = 300)
os.chdir(owd)
plt.show()

f.savefig("profile_deviations.pdf", bbox_inches='tight')
f_H_alpha=(np.max(curve)-np.min(curve))*1000
print('f_H_alpha=',round(f_H_alpha,1),'um')

#reset values
for i in range(len(sideMA[test])):
    sideMA[test][i]=(sideMA[test][i])/1000

#%%
# Calculations on right flanks
#XXXXXXXXXXXXX F_alpha XXXXXXXXXXXXXX
F_alpha_list_right=[]
for i in range(len(dist_MA_right)):
    F_alpha_list_right.append((np.max(dist_MA_right[i])-np.min(dist_MA_right[i]))*1000)

A_right=[]
k_right=[]
n_right=[]
for i in range(N_teeth):
    A=np.vstack([x_plot_right[i], np.ones(len(x_plot_right[i]))]).T
    A_right.append(A)   #Calculation of line parameters
    k, n = np.linalg.lstsq(A, dist_MA_right[i])[0]
    k_right.append(k)
    n_right.append(n)

curve=[[] for x in range(N_teeth)]
for i in range(N_teeth):
    for j in range(len(x_plot_right[i])):
        curve[i].append(k_right[i]*x_plot_right[i][j]+n_right[i])
        
distance=[[] for x in range(N_teeth)]
for i in range(N_teeth):
    for j in range(len(dist_MA_right[i])):
        distance[i].append((n_right[i]+k_right[i]*x_plot_right[i][j]-dist_MA_right[i][j])/(np.sqrt(1+k_right[i]*k_right[i])))

#curve up    
dist_ind_min=0
n_up=[]
for i in range(N_teeth):
    dist_ind_min=(np.argmin(distance[i]))
    n_up.append(dist_MA_right[i][dist_ind_min]-k_right[i]*x_plot_right[i][dist_ind_min])
curve_up=[[] for x in range(N_teeth)]
for i in range(N_teeth):
    for j in range(len(x_plot_right[i])):
        curve_up[i].append(k_right[i]*x_plot_right[i][j]+n_up[i])
#curve down        
dist_ind_max=0
n_down=[]
for i in range(N_teeth):
    dist_ind_max=(np.argmax(distance[i]))
    n_down.append(dist_MA_right[i][dist_ind_max]-k_right[i]*x_plot_right[i][dist_ind_max])
curve_down=[[] for x in range(N_teeth)]
for i in range(N_teeth):
    for j in range(len(x_plot_right[i])):
        curve_down[i].append(k_right[i]*x_plot_right[i][j]+n_down[i])

#XXXXXXXXXXXXX f_f_alpha XXXXXXXXXXXXXX
f_f_alpha_right=[]
for i in range(N_teeth):    
    f_f_alpha_right.append((n_up[i]-n_down[i])*1000)

#XXXXXXXXXXXXX f_H_alpha XXXXXXXXXXXXXX
f_H_alpha_right=[]
for i in range(N_teeth):  
    if k_right[i]>=0:    
        f_H_alpha_right.append((np.max(curve[i])-np.min(curve[i]))*1000)
    else:
        f_H_alpha_right.append(-(np.max(curve[i])-np.min(curve[i]))*1000)

#%%
# Calculations on left flanks
#XXXXXXXXXXXXX F_alpha XXXXXXXXXXXXXX
F_alpha_list_left=[]
for i in range(len(dist_MA_left)):
    F_alpha_list_left.append((np.max(dist_MA_left[i])-np.min(dist_MA_left[i]))*1000)

A_left=[]
k_left=[]
n_left=[]
for i in range(N_teeth):
    A=np.vstack([x_plot_left[i], np.ones(len(x_plot_left[i]))]).T
    A_left.append(A)   #Calculation of line parameters
    k, n = np.linalg.lstsq(A, dist_MA_left[i])[0]
    k_left.append(k)
    n_left.append(n)

curve=[[] for x in range(N_teeth)]
for i in range(N_teeth):
    for j in range(len(x_plot_left[i])):
        curve[i].append(k_left[i]*x_plot_left[i][j]+n_left[i])
        
distance=[[] for x in range(N_teeth)]
for i in range(N_teeth):
    for j in range(len(dist_MA_left[i])):
        distance[i].append((n_left[i]+k_left[i]*x_plot_left[i][j]-dist_MA_left[i][j])/(np.sqrt(1+k_left[i]*k_left[i])))

#curve up    
dist_ind_min=0
n_up=[]
for i in range(N_teeth):
    dist_ind_min=(np.argmin(distance[i]))
    n_up.append(dist_MA_left[i][dist_ind_min]-k_left[i]*x_plot_left[i][dist_ind_min])
curve_up=[[] for x in range(N_teeth)]
for i in range(N_teeth):
    for j in range(len(x_plot_left[i])):
        curve_up[i].append(k_left[i]*x_plot_left[i][j]+n_up[i])
#curve down        
dist_ind_max=0
n_down=[]
for i in range(N_teeth):
    dist_ind_max=(np.argmax(distance[i]))
    n_down.append(dist_MA_left[i][dist_ind_max]-k_left[i]*x_plot_left[i][dist_ind_max])
curve_down=[[] for x in range(N_teeth)]
for i in range(N_teeth):
    for j in range(len(x_plot_left[i])):
        curve_down[i].append(k_left[i]*x_plot_left[i][j]+n_down[i])

#XXXXXXXXXXXXX f_f_alpha XXXXXXXXXXXXXX
f_f_alpha_left=[]
for i in range(N_teeth):    
    f_f_alpha_left.append((n_up[i]-n_down[i])*1000)

#XXXXXXXXXXXXX f_H_alpha XXXXXXXXXXXXXX
f_H_alpha_left=[]
for i in range(N_teeth):   
    if k_right[i]>=0:    
        f_H_alpha_left.append((np.max(curve[i])-np.min(curve[i]))*1000)
    else:
        f_H_alpha_left.append(-(np.max(curve[i])-np.min(curve[i]))*1000)

t = PrettyTable()
t.field_names = ['Tooth', '$F_alpha_right$','$F_alpha_left$','$f_{f_alpha_right}$','$f_{f_alpha_left}$','$f_{H_alpha_right}$','$f_{H_alpha_left}$']
for i in range(N_teeth):
    t.add_row([i, round(F_alpha_list_right[i],1),round(F_alpha_list_left[i],1),
               round(f_f_alpha_right[i],1),round(f_f_alpha_left[i],1),
               round(f_H_alpha_right[i],1),round(f_H_alpha_left[i],1)])
print (t)


#%%
#============================================================================================================
#Runout evaluation
ideal_ball_pos=19.878243887764054 #FROM CAD DATA Temporary
#============================================================================================================
#First we need to determine the radius of the ball.

# =============================================================================
[n1,v1]=min(enumerate(r_cutoff_left[1]), key=lambda x: abs(x[1]-r_pc)) #calculate the values closest to the pitch circle
[n2,v2]=min(enumerate(r_cutoff_right[2]), key=lambda x: abs(x[1]-r_pc))
dist_12=np.sqrt((X_cutoff_left[1][n1]-X_cutoff_right[2][n2])**2+(Y_cutoff_left[1][n1]-Y_cutoff_right[2][n2])**2)

angleMT2=np.arctan((Y_cutoff_right[2][n2]-y_T_right[2][n2])/(X_cutoff_right[2][n2]-x_T_right[2][n2])) #angle of line perpendicular to tangent on the involute

x_circle=0
y_circle=0
distance=0
radius_circle=0
diff=5
while diff>dist_12/1000:
    x_circle=X_cutoff_right[2][n2]+np.cos(np.abs(angleMT2))*radius_circle
    y_circle=Y_cutoff_right[2][n2]+np.sin(angleMT2)*radius_circle
    distance=np.sqrt((x_circle-X_cutoff_left[1][n1])**2+(y_circle-Y_cutoff_left[1][n1])**2)
    diff=distance-radius_circle
    radius_circle=radius_circle+0.001
radius_circle=round(radius_circle,1) #Rount to the first decimal
radius_circle=1 #temporary
#Set radius for probing ball
# =============================================================================
# plt.plot(X_cutoff_left[1],Y_cutoff_left[1],marker='o',color='black', markersize=1, linestyle=' ',label="Gear flank")
# plt.plot(X_cutoff_right[2],Y_cutoff_right[2],marker='o',color='black', markersize=1, linestyle=' ')

# plt.plot(X_cutoff_left[1][n1],Y_cutoff_left[1][n1],marker='x',color='blue', markersize=3, linestyle=' ')
# plt.plot(X_cutoff_right[2][n2],Y_cutoff_right[2][n2],marker='x',color='blue', markersize=3, linestyle=' ')
# plt.plot(x_circle,y_circle,marker='x',color='blue', markersize=3, linestyle=' ',label="Probing ball")
# for_circ=np.arange(0,np.pi*2,0.1)
# x_for_circ=radius_circle*np.cos(for_circ)+x_circle
# y_for_circ=radius_circle*np.sin(for_circ)+y_circle    
# for_circ2=np.arange(np.pi/10,np.pi/5,0.1)
# x_for_circ2=r_pc*np.cos(for_circ2)
# y_for_circ2=r_pc*np.sin(for_circ2)   
# plt.plot(x_for_circ2,y_for_circ2, color='red', linestyle='-',label="Pitch circle")
# # #plt.plot(0, 0,'ro',label='Origin C.S.')
# plt.plot(x_for_circ,y_for_circ, color='blue', linestyle='-')
# plt.title('Position of probing ball',fontsize=15,**csfont)
# plt.xlabel('X [mm]',fontsize=15,**csfont)
# plt.ylabel('Y [mm]',fontsize=15,**csfont)
# plt.axis('equal')
# plt.grid()

#os.chdir(gearwd)
#
#font = font_manager.FontProperties(family='Times New Roman',
#                                   style='normal', size=16)
#
#plt.legend(prop=font)
#plt.savefig("Probing ball.png", dpi = 150)
#os.chdir(owd)
# plt.show()
# plt.legend()
#plt.show()
# =============================================================================
# =============================================================================


#%%
# """
# "Trying out the offset"

# coordinatesList = []
# for i in range(len(X_cutoff_right[2])):
#     xn=X_cutoff_right[2][i]
#     yn=Y_cutoff_right[2][i]
#     coordinatesList.append([xn , yn])

# poligon=shp.Polygon(coordinatesList)
# offsetpolyout=poligon.buffer(radius_circle)  # Outward offset
# #offsetpolyin =poligon.buffer(-radius_circle)  # Inward offset

# pointspoligon = np.array(poligon.exterior)
# pointsoout = np.array(offsetpolyout.exterior)
# #pointsin = np.array(offsetpolyin.exterior)

# plt.plot(*pointspoligon.T, color='black')
# plt.plot(*pointsoout.T, color='red')
# #plt.plot(*pointsin.T, color='green')
# plt.plot(X_cutoff_right[2],Y_cutoff_right[2],marker='o')
# #plt.plot(x_new, y_new)
# plt.axis('equal')
# plt.show()
# """
#%%
coordinatesList = []
for i in range(len(X_sorted_val)):
    xn=X_sorted_val[i]
    yn=Y_sorted_val[i]
    coordinatesList.append([xn , yn])
poligon=shp.Polygon(coordinatesList)
offsetpolyout=poligon.buffer(radius_circle)   

pointspoligon = np.array(poligon.exterior)
pointsout = np.array(offsetpolyout.exterior)

# f = plt.figure()
# plt.plot(*pointspoligon.T, color='black')
# plt.plot(*pointsout.T, color='red',label='Offset of section')
# #plt.plot(X_sorted_val,Y_sorted_val,marker='o',markersize=1)
# #plt.plot(X_cutoff_left[1],Y_cutoff_left[1],marker='o',color='red', markersize=1, linestyle=' ')
# #plt.plot(X_cutoff_right[2],Y_cutoff_right[2],marker='o',color='red', markersize=1, linestyle=' ')
# #plt.plot(X_cutoff_left[1][n1],Y_cutoff_left[1][n1],marker='x',color='black', markersize=3, linestyle=' ')
# #plt.plot(X_cutoff_right[2][n2],Y_cutoff_right[2][n2],marker='x',color='black', markersize=3, linestyle=' ')
# plt.plot(x_circle,y_circle,marker='x',color='blue', markersize=4, linestyle=' ',label='Centre of probing ball')
# plt.plot(0,0,marker='x',color='black', markersize=4, linestyle=' ',label='Gear centre')
# for_circ=np.arange(0,np.pi*2,0.1)
# inner_circle=np.arange(0,np.pi*2,0.1)
# inner_radius=6.15/2
# x_inner_circle=inner_radius*np.cos(inner_circle)
# y_inner_circle=inner_radius*np.sin(inner_circle)  
# plt.plot(x_inner_circle, y_inner_circle, color='black')
# x_for_circ=radius_circle*np.cos(for_circ)+x_circle
# y_for_circ=radius_circle*np.sin(for_circ)+y_circle    
# for_circ2=np.arange(0,np.pi*2,0.1)
# x_for_circ2=r_pc*np.cos(for_circ2)
# y_for_circ2=r_pc*np.sin(for_circ2)   
# plt.plot(x_for_circ2,y_for_circ2, color='orange', linestyle='-.',label='Pitch circle')
# plt.plot(x_for_circ,y_for_circ, color='blue', linestyle='-',label='Probing ball')
# plt.axis('equal')
# #plt.title('Offset by radius of probing ball')
# plt.xlabel('$x$ [mm]',**axis_font)
# plt.ylabel('$y$ [mm]',**axis_font)
# plt.grid()
# plt.tight_layout()
# plt.legend(prop={'family':'Times new Roman', 'size':16},markerscale=2)
# #axes = plt.gca()
# #axes.set_xlim([7,11])
# #axes.set_ylim([3,6.5])
# axes = plt.gca()
# # axes.set_xlim([-2,12])
# # axes.set_ylim([-2,12])
# os.chdir(gearwd)
# plt.savefig("runout_evaluation1.png", dpi = 200)
# f.savefig("Runout.pdf", bbox_inches='tight')
# os.chdir(owd)
# plt.show()


#f = plt.figure()
plt.plot(*pointspoligon.T, color='black')
plt.plot(*pointsout.T, color='red',label='Offset of section')
#plt.plot(X_sorted_val,Y_sorted_val,marker='o',markersize=1)
#plt.plot(X_cutoff_left[1],Y_cutoff_left[1],marker='o',color='red', markersize=1, linestyle=' ')
#plt.plot(X_cutoff_right[2],Y_cutoff_right[2],marker='o',color='red', markersize=1, linestyle=' ')
plt.plot(X_cutoff_left[1][n1],Y_cutoff_left[1][n1],marker='o',color='blue', markersize=8, linestyle=' ')
plt.plot(X_cutoff_right[2][n2],Y_cutoff_right[2][n2],marker='o',color='blue', markersize=8, linestyle=' ')
plt.plot(x_circle,y_circle,marker='x',color='blue', markersize=8, linestyle=' ',label='Centre of probing ball')
for_circ=np.arange(0,np.pi*2,0.1)
x_for_circ=radius_circle*np.cos(for_circ)+x_circle
y_for_circ=radius_circle*np.sin(for_circ)+y_circle    
for_circ2=np.arange(0,np.pi*2,0.1)
x_for_circ2=r_pc*np.cos(for_circ2)
y_for_circ2=r_pc*np.sin(for_circ2)   
plt.plot(x_for_circ2,y_for_circ2, color='orange', linestyle='-.',label='Pitch circle')
plt.plot(x_for_circ,y_for_circ, color='blue', linestyle='-',label='Probing ball')
plt.axis('equal')
#plt.title('Offset by radius of probing ball')
plt.xlabel('$x$ [mm]',**axis_font)
plt.ylabel('$y$ [mm]',**axis_font)
plt.grid()
plt.tight_layout()
#plt.legend(prop={'family':'Times new Roman', 'size':16},markerscale=1)
axes = plt.gca()
axes.set_xlim([17,22])
axes.set_ylim([3,8])
os.chdir(gearwd)
plt.savefig("runout_evaluation2.png", dpi = 200)
#f.savefig("Runout.pdf", bbox_inches='tight')
os.chdir(owd)
plt.show()

#We now find the minimum for pointsout for each pair of teeth
#pointsout[0][1] - o is first point
#%%
"Identify the points for runout and plot"
r_out=[]
fi_out=[]
fi_calc=0
for i in range(len(pointsout)):
    r_out.append(np.sqrt(pointsout[i][0]**2+pointsout[i][1]**2))
    fi_calc=math.atan2(pointsout[i][1],pointsout[i][0])*180/np.pi
    if fi_calc<0:
        fi_calc=fi_calc+360
    fi_out.append(fi_calc)

r_tooth_pts=[[] for x in range(N_teeth)]
fi_tooth_pts=[[] for x in range(N_teeth)]
for i in range(N_teeth):
    for j in range(len(fi_out)):
        if fi_out[j]<fi_range*(i+1) and fi_out[j]>fi_range*(i):
            r_tooth_pts[i].append(r_out[j])    
            fi_tooth_pts[i].append(fi_out[j])

x_tooth_pts=[]
y_tooth_pts=[]
indx=0
dist_ball_center=[]
for i in range(N_teeth):
    indx=np.argmin(r_tooth_pts[i])
    x_tooth_pts.append(r_tooth_pts[i][indx]*np.cos(np.radians(fi_tooth_pts[i][indx])))
    y_tooth_pts.append(r_tooth_pts[i][indx]*np.sin(np.radians(fi_tooth_pts[i][indx])))
    dist_ball_center.append(r_tooth_pts[i][indx]) #the distance to the center, we need it for calculating

F_r=(np.max(dist_ball_center)-np.min(dist_ball_center))*1000

plt.plot(*pointspoligon.T, color='black')
plt.plot(*pointsout.T, color='red')
plt.plot(X_sorted_val,Y_sorted_val,marker='o',markersize=1)
plt.plot(x_tooth_pts,y_tooth_pts,marker='o',color='blue', markersize=3,linestyle=' ')
plt.axis('equal')
plt.title('Offset by radius of probing ball, identified points of ball centre')
plt.xlabel('x [mm]')
plt.ylabel('y [mm]')
plt.grid()
plt.show()



#%%
"Import of new offset sections."
"Temporary"
"Temporary"
"Temporary" "To je za več presekov na Fr grafu"
def forFroffset(X_offset,Y_offset,fi_range,dist_ball_center_offset,F_r_offset):
    #"Identify inner circle."
    r_offset=[]
    for i in range(len(X_offset)):
        r_offset.append(math.sqrt(X_offset[i]**2 + Y_offset[i]**2))
    
    r_cutoff_offset=radius_b*9.5/10        #define the radius for the cutoff
    circleNindex_offset=[]
    
    for i in range(len(r_offset)):
        if r_offset[i] <r_cutoff_offset:
            circleNindex_offset.append(i) #a set of indexes for the inner circle
    
    "Remove the inner circle values"
    Xinner_offset=[]
    Yinner_offset=[]
    c=0
    for i in circleNindex_offset:
        Xinner_offset.append(X_offset[i-c])
        Yinner_offset.append(Y_offset[i-c])
        X_offset.pop(i-c)
        Y_offset.pop(i-c)
        c=c+1
    
    r_offset=[]
    fi_offset=[]
    angle_offset=0
    for i in range(len(X_offset)):
        r_offset.append(math.sqrt(X_offset[i]**2 + Y_offset[i]**2))
        angle_offset=math.atan2(Y_offset[i],X_offset[i])*(180/np.pi) #angle calculation, return 0<fi<360
        if angle_offset<0:
            angle_offset=angle_offset+360
        fi_offset.append(angle_offset)
    
    "Sort the points by angle fi"
    fi_r_sorted_offset=sorted(zip(fi_offset,r_offset))
    r_sorted_offset=[r_offset for fi_offset, r_offset in fi_r_sorted_offset]
    fi_sorted_offset=sorted(fi_offset)
    
    fi_sorted_val_offset=[]
    for i in range(len(fi_sorted_offset)):
        fi_sorted_val_offset.append(fi_sorted_offset[i]*np.pi/180)
    X_sorted_val_offset=[]
    Y_sorted_val_offset=[]
    X_sorted_val_offset=r_sorted_offset*np.cos(fi_sorted_val_offset)    #need angle in radians
    Y_sorted_val_offset=r_sorted_offset*np.sin(fi_sorted_val_offset)    #need angle in radians
    
    
    coordinatesList_offset = []
    for i in range(len(X_sorted_val_offset)):
        xn=X_sorted_val_offset[i]
        yn=Y_sorted_val_offset[i]
        coordinatesList_offset.append([xn , yn])
    poligon_offset=shp.Polygon(coordinatesList_offset)
    offsetpolyout_offset=poligon_offset.buffer(radius_circle)   
    
    pointspoligon_offset = np.array(poligon_offset.exterior)
    pointsout_offset = np.array(offsetpolyout_offset.exterior)
    
    r_out_offset=[]
    fi_out_offset=[]
    fi_calc=0
    for i in range(len(pointsout_offset)):
        r_out_offset.append(np.sqrt(pointsout_offset[i][0]**2+pointsout_offset[i][1]**2))
        fi_calc=math.atan2(pointsout_offset[i][1],pointsout_offset[i][0])*180/np.pi
        if fi_calc<0:
            fi_calc=fi_calc+360
        fi_out_offset.append(fi_calc)
    
    r_tooth_pts_offset=[[] for x in range(N_teeth)]
    fi_tooth_pts_offset=[[] for x in range(N_teeth)]
    for i in range(N_teeth):
        for j in range(len(fi_out_offset)):
            if fi_out_offset[j]<fi_range*(i+1) and fi_out_offset[j]>fi_range*(i):
                r_tooth_pts_offset[i].append(r_out_offset[j])    
                fi_tooth_pts_offset[i].append(fi_out_offset[j])
    
    x_tooth_pts_offset=[]
    y_tooth_pts_offset=[]
    indx=0
    dist_ball_center_offset=[]
    for i in range(N_teeth):
        indx=np.argmin(r_tooth_pts_offset[i])
        x_tooth_pts_offset.append(r_tooth_pts_offset[i][indx]*np.cos(np.radians(fi_tooth_pts_offset[i][indx])))
        y_tooth_pts_offset.append(r_tooth_pts_offset[i][indx]*np.sin(np.radians(fi_tooth_pts_offset[i][indx])))
        dist_ball_center_offset.append(r_tooth_pts_offset[i][indx]) #the distance to the center, we need it for calculating
    
    F_r_offset=(np.max(dist_ball_center_offset)-np.min(dist_ball_center_offset))*1000
    return (dist_ball_center_offset,F_r_offset)


# "Nov file"

# folder='Novi_scani_10_12_2020\\obdelava\\1_10\\1-10_3_geometric_alignment'
# X_offset_1_5=[]
# Y_offset_1_5=[]

# owd=os.getcwd()
# os.chdir(folder)
# gearwd=os.getcwd()
# file = open('Plane Z +0.000 mm-1-5.asc',"r")    #planar section
# os.chdir(owd)

# for line in file:
#     x, y, z = line.split()
#     X_offset_1_5.append(float(x))                      #need to assign float
#     Y_offset_1_5.append(float(y))
    
# dist_ball_center_offset_1_5=[]
# F_r_offset_1_5=0
# [dist_ball_center_offset_1_5,F_r_offset_1_5]=forFroffset(X_offset_1_5,Y_offset_1_5,fi_range,dist_ball_center_offset_1_5,F_r_offset_1_5)

# "Nov file 2"
# folder='Novi_scani_10_12_2020\\obdelava\\1_10\\1-10_3_geometric_alignment'
# X_offset_plus_1_5=[]
# Y_offset_plus_1_5=[]

# owd=os.getcwd()
# os.chdir(folder)
# gearwd=os.getcwd()
# file = open('Plane Z +0.000 mm+1-5.asc',"r")    #planar section
# os.chdir(owd)

# for line in file:
#     x, y, z = line.split()
#     X_offset_plus_1_5.append(float(x))                      #need to assign float
#     Y_offset_plus_1_5.append(float(y))
    
# dist_ball_center_offset_plus_1_5=[]
# F_r_offset_plus_1_5=0
# [dist_ball_center_offset_plus_1_5,F_r_offset_plus_1_5]=forFroffset(X_offset_plus_1_5,Y_offset_plus_1_5,fi_range,dist_ball_center_offset_plus_1_5,F_r_offset_plus_1_5)
# "Temporary"
# "Temporary"
# "Temporary"
# "Temporary"
# "Temporary"

#%%
"Fr statistical analisys"

average_ball_distance=np.average(dist_ball_center)

relative_ball_dev= (average_ball_distance-ideal_ball_pos)/ideal_ball_pos*100
print('Relative deviation is:',round(relative_ball_dev,3),'%')

st_dev_ball=np.std(dist_ball_center)*1000

#Fi=(np.abs(np.max(dist_ball_center)-ideal_ball_pos)  - np.abs(ideal_ball_pos-np.min(dist_ball_center)))*1000   #in microns

Fi=(np.min(dist_ball_center)-ideal_ball_pos  +   np.max(dist_ball_center)-ideal_ball_pos  )/2         *1000
#%%
#dist_ball_center.append(dist_ball_center[0]) #We add one more point, so that the last point = first
x_for_plot=range(1,N_teeth+1)
f = plt.figure()
plt.plot(x_for_plot,dist_ball_center,marker='o',markersize=4,color='black')
plt.plot(x_for_plot,dist_ball_center,color='black',label="srednji profil")
#plt.axis('equal')

# plt.plot(x_for_plot,dist_ball_center_offset_1_5,marker='o',markersize=4,color='red') #temporary  not inclued
# plt.plot(x_for_plot,dist_ball_center_offset_1_5,color='red',label="minus 1.5 mm")                        #temporary  not inclued

# plt.plot(x_for_plot,dist_ball_center_offset_plus_1_5,marker='o',markersize=4,color='blue') #temporary  not inclued
# plt.plot(x_for_plot,dist_ball_center_offset_plus_1_5,color='blue',label="plus 1.5 mm")                        #temporary not inclued
# plt.legend()

plt.title('Runout evaluation')
plt.xlabel('The index of the gap between teeth')
plt.ylabel('The offset of the measuring ball [mm]')
axes = plt.gca()
#axes.set_ylim([np.min(dist_ball_center),np.max(dist_ball_center)])
#plt.axhline(y=ideal_ball_pos, color='k', linestyle='--',linewidth=1,label="Ideal")      #temporary
#plt.text(0, ideal_ball_pos, round(ideal_ball_pos,3),**axis_font)                         #temporary
#plt.text(0, ideal_ball_pos, round(ideal_ball_pos,3),**axis_font)
#plt.text(0, ideal_ball_pos-0.01, "Ideal",**axis_font)
##
#plt.axhline(y=average_ball_distance, color='r', linestyle='--',linewidth=1,label="Average") #temporary
#plt.text(0, average_ball_distance, round(average_ball_distance,3),**axis_font)       #temporary
#plt.text(0, ideal_ball_pos, round(ideal_ball_pos,3),**axis_font)
#plt.text(0, average_ball_distance-0.01, "Average",**axis_font)

#plt.text(N_teeth-3, average_ball_distance+0.01, round(st_dev_ball,3),**axis_font)   #temporary
#plt.text(0, ideal_ball_pos, round(ideal_ball_pos,3),**axis_font)
#plt.text(N_teeth-3, average_ball_distance-0.01, round(relative_ball_dev,3),**axis_font)      #temporary
#plt.text(0, ideal_ball_pos, round(ideal_ball_pos,3),**axis_font)
plt.grid()
plt.tight_layout()
os.chdir(gearwd)
plt.savefig("F_r.png", dpi = 150)
f.savefig("Runout.pdf", bbox_inches='tight')
os.chdir(owd)
plt.show()

print('F_r=',round(F_r,1),'um')

#%%
"#Evaluate the tolerances. First we determine the width and the diameter of the"
#hole. Then we read the tolerances from the STEP file.

#determine the hole diameter
r_inner=[]
for i in range(len(Xinner)):
    r_inner.append(math.sqrt(Xinner[i]**2 + Yinner[i]**2))

d_inner=np.average(r_inner)*2
print('The hole diameter is:',round(d_inner,3),'mm')

#determine the gear width
Z_flank_divided_max=[]
Z_flank_divided_min=[]
Z_flank_distance=[]
for i in range(len(Z_flank_divided)):
    Z_flank_divided_max.append(np.max(Z_flank_divided[i]))
    Z_flank_divided_min.append(np.min(Z_flank_divided[i]))
    Z_flank_distance.append(np.max(Z_flank_divided[i])-np.min(Z_flank_divided[i]))  #the width of the gear calculated on each tooth

gear_width=0
gear_width=np.average(Z_flank_distance) #average width of gear
print('The gear width is:',round(gear_width,3),'mm')

gear_width_Q=np.round(gear_width) #will be used to calculate the gear Q number for parameters
print('Nominal gear width is:',gear_width_Q,'mm')

# =============================================================================
#%%
# #Read the tolerances from the STEP file
# dim_char_rep=[]
# hash1=[]
# hash2=[]
# end1=0
# end2=0
# end1=0
# end2=0
# num1=0
# num2=0
# character='#'
# searchfile = open("Testiranje MBD\\1\\ZOBNIK_PROFIL_C_MBD.stp", "r")
# for line in searchfile:
#     if "DIMENSIONAL_CHARACTERISTIC_REPRESENTATION" in line:
#         #print(line)
#         dim_char_rep.append(line)
# searchfile.close()
# #find the number for determining if it is diameter or linear distance
# hash1=[pos for pos, char in enumerate(dim_char_rep[0]) if char==character]
# hash2=[pos for pos, char in enumerate(dim_char_rep[1]) if char==character]
# end1=dim_char_rep[0].find(',')
# end2=dim_char_rep[1].find(',')
# 
# num1=dim_char_rep[0][hash1[1]:end1]+'='
# num2=dim_char_rep[1][hash2[1]:end2]+'='
# 
# #now we can read the second line
# num1_find=0
# num2_find=0
# searchfile = open("Testiranje MBD\\1\\ZOBNIK_PROFIL_C_MBD.stp", "r")
# for line in searchfile:
#     if num1 in line:
#         #print(line)
#         num1_find=line
#     if num2 in line:
#         #print(line)
#         num2_find=line
# searchfile.close()
# 
# #determine if diameter or linear distance -> 1=diameter, 2=linear distance
# type_dim_1=0
# if "diameter" in num1_find:
#     type_dim_1=1
# if "linear distance" in num1_find:
#     type_dim_1=2   
# type_dim_2=0
# if "diameter" in num2_find:
#     type_dim_2=1
# if "linear distance" in num2_find:
#     type_dim_2=2  
# 
# #now we search for the second number in the lines
# end21=0
# end22=0
# num21=0
# num22=0
# end21=dim_char_rep[0].find(')')
# end22=dim_char_rep[1].find(')')
# num21=dim_char_rep[0][hash1[2]:end21]+'='
# num22=dim_char_rep[1][hash2[2]:end22]+'='
# 
# num21_find=0
# num22_find=0
# searchfile = open("Testiranje MBD\\1\\ZOBNIK_PROFIL_C_MBD.stp", "r")
# for line in searchfile:
#     if num21 in line:
#         #print(line)
#         num21_find=line
#     if num22 in line:
#         #print(line)
#         num22_find=line
# searchfile.close()
# 
# #select the number
# hash321=[]
# hash322=[]
# end321=0
# end322=0
# hash321=[pos for pos, char in enumerate(num21_find) if char==character]
# hash322=[pos for pos, char in enumerate(num22_find) if char==character]
# end321=num21_find.find(')')
# end322=num22_find.find(')')
# 
# num321=0
# num322=0    
# num321=num21_find[hash321[1]:end321]+'='
# num322=num22_find[hash322[1]:end322]+'='
# 
# num321_find=0
# num322_find=0
# searchfile = open("Testiranje MBD\\1\\ZOBNIK_PROFIL_C_MBD.stp", "r")
# for line in searchfile:
#     if num321 in line:
#         #print(line)
#         num321_find=line
#     if num322 in line:
#         #print(line)
#         num322_find=line
# searchfile.close()
# 
# start4321=[]
# start4322=[]
# end4321=[]
# end4322=[]
# start4321=[pos for pos, char in enumerate(num321_find) if char=='(']
# start4322=[pos for pos, char in enumerate(num322_find) if char=='(']
# end4321=[pos for pos, char in enumerate(num321_find) if char==')']
# end4322=[pos for pos, char in enumerate(num322_find) if char==')']
# 
# # Here we get the actual dimensions, the next step is getting the plus/minus tolerances
# measure1=0
# measure2=0
# measure1=float(num321_find[start4321[4]+1:end4321[2]])
# measure2=float(num322_find[start4322[4]+1:end4322[2]])
# 
# #getting the tolerances
# tolerance_lines=[]
# searchfile = open("Testiranje MBD\\1\\ZOBNIK_PROFIL_C_MBD.stp", "r")
# for line in searchfile:
#     if "PLUS_MINUS_TOLERANCE" in line:
#         #print(line)
#         tolerance_lines.append(line)
# searchfile.close()
# 
# tolhash1=[pos for pos, char in enumerate(tolerance_lines[0]) if char==character]
# tolhash2=[pos for pos, char in enumerate(tolerance_lines[1]) if char==character]
# tolend1=tolerance_lines[0].find(',')
# tolend2=tolerance_lines[1].find(',')
# tolend21=tolerance_lines[0].find(')')
# tolend22=tolerance_lines[1].find(')')
# 
# tolnum1=tolerance_lines[0][tolhash1[1]:tolend1]+'='
# tolnum2=tolerance_lines[1][tolhash2[1]:tolend2]+'='
# 
# #If we need to check if they are both for the diamater or linear distance
# tolnum21=tolerance_lines[0][tolhash1[2]:tolend21]+'='
# tolnum22=tolerance_lines[1][tolhash2[2]:tolend22]+'='
# 
# 
# #Determine the tolerance value
# tolnum1_find=0
# tolnum2_find=0
# searchfile = open("Testiranje MBD\\1\\ZOBNIK_PROFIL_C_MBD.stp", "r")
# for line in searchfile:
#     if tolnum1 in line:
#         #print(line)
#         tolnum1_find=line
#     if tolnum2 in line:
#         #print(line)
#         tolnum2_find=line
# searchfile.close()
# 
# tolhash1_find=[]
# tolhash2_find=[]
# tolend1_find=0
# tolend2_find=0
# tolend21_find=0
# tolend22_find=0
# tolhash1_find=[pos for pos, char in enumerate(tolnum1_find) if char==character]
# tolend1_find=tolnum1_find.find(',')
# tolend2_find=tolnum1_find.find(')')
#     
# tolhash2_find=[pos for pos, char in enumerate(tolnum2_find) if char==character]
# tolend21_find=tolnum2_find.find(',')
# tolend22_find=tolnum2_find.find(')')
# 
# tolvalnum1=0
# tolvalnum2=0
# tolvalnum21=0
# tolvalnum22=0
# tolvalnum1=tolnum1_find[tolhash1_find[1]:tolend1_find]+'=' #this is the lower bound
# tolvalnum2=tolnum1_find[tolhash1_find[2]:tolend2_find]+'=' #the upper bound of the tolerance
# 
# tolvalnum21=tolnum2_find[tolhash2_find[1]:tolend21_find]+'='
# tolvalnum22=tolnum2_find[tolhash2_find[2]:tolend22_find]+'='
# 
# tolerance_value1=0
# tolerance_value2=0
# tolerance_value21=0
# tolerance_value22=0
# searchfile = open("Testiranje MBD\\1\\ZOBNIK_PROFIL_C_MBD.stp", "r")
# for line in searchfile:
#     if tolvalnum1 in line:
#         #print(line)
#         tolerance_value1=line
#     if tolvalnum2 in line:
#         #print(line)
#         tolerance_value2=line
#     if tolvalnum21 in line:
#         #print(line)
#         tolerance_value21=line
#     if tolvalnum22 in line:
#         #print(line)
#         tolerance_value22=line
# searchfile.close()
# 
# 
# #find the tolerance values for the first dimension
# tolvalue1_start=[pos for pos, char in enumerate(tolerance_value1) if char=='(']
# tolvalue1_end=tolerance_value1.find(')')
# 
# tolvalue2_start=[pos for pos, char in enumerate(tolerance_value2) if char=='(']
# tolvalue2_end=tolerance_value2.find(')')
# 
# tolvalue21_start=[pos for pos, char in enumerate(tolerance_value21) if char=='(']
# tolvalue21_end=tolerance_value21.find(')')
# 
# tolvalue22_start=[pos for pos, char in enumerate(tolerance_value22) if char=='(']
# tolvalue22_end=tolerance_value22.find(')')
# 
# #Here are our tolerances
# tol1lower=float(tolerance_value1[tolvalue1_start[1]+1:tolvalue1_end])
# tol1upper=float(tolerance_value2[tolvalue2_start[1]+1:tolvalue2_end])
# 
# tol2lower=float(tolerance_value21[tolvalue21_start[1]+1:tolvalue21_end])
# tol2upper=float(tolerance_value22[tolvalue22_start[1]+1:tolvalue22_end])
# 
# #Now we need to gather the values
# #type_dim -> 1=diameter, 2=linear
# print('First tolerance is of type',type_dim_1,'. The value is:', measure1,'. The tolerances are:',tol1lower,'to',tol1upper,'. Measures in milimeters.')
# print('Second tolerance is of type',type_dim_2,'. The value is:', measure2,'. The tolerances are:',tol2lower,'to',tol2upper,'. Measures in milimeters.')
# 
# #Now lets evaluate the measured dimensions
# value1_up=0
# value1_down=0
# value2_up=0
# value2_down=0
# 
# value1_up=measure1+tol1upper
# value1_down=measure1+tol1lower
# value2_up=measure2+tol2upper
# value2_down=measure2+tol2lower
# 
# dinbounds=0
# winbounds=0
# if type_dim_1==1:
#     if d_inner>value1_down and d_inner<value1_up:
#         dinbounds='Yes'
#     else:
#         dinbounds='No'
# elif type_dim_1==2:
#     if gear_width>value1_down and gear_width<value1_up:
#         winbounds='Yes'
#     else:
#         winbounds='No'
#         
# if type_dim_2==1:
#     if d_inner>value2_down and d_inner<value2_up:
#         dinbounds='Yes'
#     else:
#         dinbounds='No'
# elif type_dim_2==2:
#     if gear_width>value2_down and gear_width<value2_up:
#         winbounds='Yes'
#     else:
#         winbounds='No'
# 
# print('Is the gear hole within tolerances?',dinbounds)
# print('Is the gear width within tolerances?',winbounds)
# 
# =============================================================================
#%%

#Here we calculate in which tolerance grade the parameters are.

#We first need to determine the geometric value of the range
d_borders=[5,20,50,125,280,560,1000,1600,2500,4000,6000,8000,10000]
m_borders=[0.5,2,3.5,6,10,16,25,40,70]
b_borders=[4,10,20,40,80,160,250,400,650,1000]
d_Q=0
m_Q=0
b_Q=0

for i in range(len(d_borders)):
    if pitch_circle>d_borders[i] and pitch_circle<=d_borders[i+1]:
        d_Q=np.sqrt(d_borders[i]*d_borders[i+1])

#!!!!!
#d_Q=np.sqrt(20*50)
#!!!!!
     
for i in range(len(m_borders)):
    if module>m_borders[i] and module<=m_borders[i+1]:
        m_Q=np.sqrt(m_borders[i]*m_borders[i+1])

for i in range(len(b_borders)):
    if gear_width_Q>b_borders[i] and gear_width_Q<=b_borders[i+1]:
        b_Q=np.sqrt(b_borders[i]*b_borders[i+1])
        
# DIN VALUES ==================================================================
m_DIN_borders=[1,2,3.55]
d_DIN_borders=[10,50,125,280,560,1000,1600,2500,4000,6300,10000] # parting circle diameter [mm]
b_DIN_borders=[20,40,100,160]

b_Q_DIN=0

for i in range(len(m_DIN_borders)):
    if module>=m_DIN_borders[i] and module<=m_DIN_borders[i+1]:
        m_Q_DIN=np.sqrt(m_DIN_borders[i]*m_DIN_borders[i+1])
m_Q_DIN=0.5         #vprasljivo

for i in range(len(d_DIN_borders)):
    if pitch_circle>d_DIN_borders[i] and pitch_circle<=d_DIN_borders[i+1]:
        d_Q_DIN=np.sqrt(d_DIN_borders[i]*d_DIN_borders[i+1])

for i in range(len(b_DIN_borders)):
    if gear_width<b_DIN_borders[0]:
        b_Q_DIN=14         #vprasljivo
    elif gear_width_Q>b_DIN_borders[i] and gear_width_Q<=b_DIN_borders[i+1]:
        b_Q_DIN=np.sqrt(b_DIN_borders[i]*b_DIN_borders[i+1])
        
#b_Q_DIN=b_Q
#==============================================================================
#function for rounding to half an integer
def round_half(number):
    return round(number * 2) / 2

#Functions for rounding
def rounding_function(parameterQrange):
    for i in range(len(parameterQrange)):
        if parameterQrange[i]>10:
            parameterQrange[i]=np.round(parameterQrange[i])
        if parameterQrange[i]<10 and parameterQrange[i]>5:
            parameterQrange[i]=round_half(parameterQrange[i])
        if parameterQrange[i]<5:
            parameterQrange[i]=round(parameterQrange[i],1)
    return()
#==============================================================================
#Function to find grade
def findGrade(value_list,max_value,parameterQrange,parameterQvalue):
    max_value=np.max(np.abs(value_list))
    parameterQvalue=13 #every value that is bigger than the 12th grade is 13
    for i in range(0,12):
        if max_value>parameterQrange[i] and max_value<parameterQrange[i+1]:
            parameterQvalue=i+1
    if max_value>=0 and max_value<parameterQrange[0]:
        parameterQvalue=0
    return (parameterQvalue)
#==============================================================================
#Start with f_{pt}
f_pt_Q5=0
f_pt_Q5=0.3*(m_Q+0.4*np.sqrt(d_Q))+4

f_pt_Qrange=[]
for i in range(0,13):
    f_pt_Qrange.append(f_pt_Q5*math.pow(2,0.5*(i-5)))
rounding_function(f_pt_Qrange) #round the numbers

"""
#We need to round the numbers
for i in range(len(f_pt_Qrange)):
    if f_pt_Qrange[i]>10:
        f_pt_Qrange[i]=np.round(f_pt_Qrange[i])
    if f_pt_Qrange[i]<10 and f_pt_Qrange[i]>5:
        f_pt_Qrange[i]=round_half(f_pt_Qrange[i])
    if f_pt_Qrange[i]<5:
        f_pt_Qrange[i]=round(f_pt_Qrange[i],1)
f_pt_left_max=np.max(np.abs(f_pt_left))
f_pt_right_max=np.max(np.abs(f_pt_right))
for i in range(0,12):
    if f_pt_left_max>f_pt_Qrange[i] and f_pt_left_max<f_pt_Qrange[i+1]:
        f_pt_left_Q=i+1
    if f_pt_right_max>f_pt_Qrange[i] and f_pt_right_max<f_pt_Qrange[i+1]:
        f_pt_right_Q=i+1
"""
f_pt_left_max=0
f_pt_left_Q=0
f_pt_right_max=0
f_pt_right_Q=0
f_pt_left_Q=findGrade(f_pt_left,f_pt_left_max,f_pt_Qrange,f_pt_left_Q)
f_pt_right_Q=findGrade(f_pt_right,f_pt_right_max,f_pt_Qrange,f_pt_right_Q)
print('Parameter f_pt_left is in ISO grade',f_pt_left_Q)
print('Parameter f_pt_right is in ISO grade',f_pt_right_Q)

#DIN DIN DIN DIN
f_pt_DIN_Q5=0
f_pt_DIN_Q5=4+0.315*(m_Q_DIN+0.25*np.sqrt(d_Q_DIN))

f_pt_DIN_Qrange=[]
for i in range(0,10):  #0-9
    f_pt_DIN_Qrange.append(f_pt_DIN_Q5*math.pow(1.4,i-5))
f_pt_DIN_Q9=f_pt_DIN_Qrange[-1]
for i in range(10,13):
    f_pt_DIN_Qrange.append(f_pt_DIN_Q9*math.pow(1.6,i-9)) #for 10-12
rounding_function(f_pt_DIN_Qrange) #round the numbers

###############################################################################
#temporary  for modules 1-2 and reference circle 10-50
f_pt_DIN_Qrange=[]
f_pt_DIN_Qrange=[0.5,1,1.5,2.5,3.5,5,7,9,14,18,28,50,80]

##############################################################################
f_pt_left_Q_DIN=0
f_pt_right_Q_DIN=0
f_pt_left_Q_DIN=findGrade(f_pt_left,f_pt_left_max,f_pt_DIN_Qrange,f_pt_left_Q_DIN)
f_pt_right_Q_DIN=findGrade(f_pt_right,f_pt_right_max,f_pt_DIN_Qrange,f_pt_right_Q_DIN)
#DIN DIN DIN DIN

#==============================================================================
#F_pk
F_pk_Q5=0
F_pk_Q5=f_pt_Q5+1.6*np.sqrt((k_eval_teeth-1)*m_Q)

F_pk_Qrange=[]
for i in range(0,13):
    F_pk_Qrange.append(F_pk_Q5*math.pow(2,0.5*(i-5)))
rounding_function(F_pk_Qrange) #round the numbers

F_pk_left_max=0
F_pk_left_Q=0
F_pk_right_max=0
F_pk_right_Q=0
F_pk_left_Q=findGrade(Fpk_max_diff_left,F_pk_left_max,F_pk_Qrange,F_pk_left_Q)
F_pk_right_Q=findGrade(Fpk_max_diff_right,F_pk_right_max,F_pk_Qrange,F_pk_right_Q)
print('Parameter F_p_',k_eval_teeth,'_left is in ISO grade',F_pk_left_Q)
print('Parameter F_p_',k_eval_teeth,'_right is in ISO grade',F_pk_right_Q)

#==============================================================================
#F_p_Q5 parameter
F_p_Q5=0
F_p_Q5=0.3*m_Q+1.25*np.sqrt(d_Q)+7

F_p_Qrange=[]
for i in range(0,13):
    F_p_Qrange.append(F_p_Q5*math.pow(2,0.5*(i-5)))
rounding_function(F_p_Qrange)

F_p_left_max=0
F_p_left_Q=0
F_p_right_max=0
F_p_right_Q=0
F_p_left_Q=findGrade(F_p_left,F_p_left_max,F_p_Qrange,F_p_left_Q)
F_p_right_Q=findGrade(F_p_right,F_p_right_max,F_p_Qrange,F_p_right_Q)
print('Parameter F_p_left is in ISO grade',F_p_left_Q)
print('Parameter F_p_right is in ISO grade',F_p_right_Q)

#DIN DIN DIN DIN
F_p_DIN_Q5=0
F_p_DIN_Q5=7.25*((d_Q_DIN**(1/3))/(N_teeth**(1/7)))

F_p_DIN_Qrange=[]
for i in range(0,10):  #0-4
    F_p_DIN_Qrange.append(F_p_DIN_Q5*math.pow(1.4,i-5)) #for 0-9
F_p_DIN_Q9=F_p_DIN_Qrange[-1]
for i in range(10,13):
    F_p_DIN_Qrange.append(F_p_DIN_Q9*math.pow(1.6,i-9)) #for 10-12
rounding_function(F_p_DIN_Qrange) #round the numbers
###############################################################################
#temporary
F_p_DIN_Qrange=[]
F_p_DIN_Qrange=[0.5,3.5,5,7,10,14,18,28,36,50,80,140,220]
###############################################################################
F_p_left_Q_DIN=0
F_p_left_max=0
F_p_right_Q_DIN=0
F_p_right_max=0
F_p_left_Q_DIN=findGrade(F_p_left,F_p_left_max,F_p_DIN_Qrange,F_p_left_Q_DIN)
F_p_right_Q_DIN=findGrade(F_p_right,F_p_right_max,F_p_DIN_Qrange,F_p_right_Q_DIN)
#DIN DIN DIN DIN

#==============================================================================
#F_f_alpha_Q5 parameter
F_f_alpha_Q5=0
F_f_alpha_Q5=2.5*np.sqrt(m_Q)+0.17*np.sqrt(d_Q)+0.5

F_f_alpha_Qrange=[]
for i in range(0,13):
    F_f_alpha_Qrange.append(F_f_alpha_Q5*math.pow(2,0.5*(i-5)))
rounding_function(F_f_alpha_Qrange)

F_f_alpha_left_max=0
F_f_alpha_left_Q=0
F_f_alpha_right_max=0
F_f_alpha_right_Q=0
f_f_alpha_left_Q=findGrade(f_f_alpha_left,F_f_alpha_left_max,F_f_alpha_Qrange,F_f_alpha_left_Q)
f_f_alpha_right_Q=findGrade(f_f_alpha_right,F_f_alpha_right_max,F_f_alpha_Qrange,F_f_alpha_right_Q)
print('Parameter f_f_alpha_left is in ISO grade',f_f_alpha_left_Q)
print('Parameter f_f_alpha_right is in ISO grade',f_f_alpha_right_Q)

#DIN DIN DIN DIN
F_f_alpha_DIN_Q5=0
F_f_alpha_DIN_Q5=1.5+0.25*(m_Q_DIN+9*np.sqrt(m_Q_DIN))

F_f_alpha_DIN_Qrange=[]
for i in range(0,10):  #0-9
    F_f_alpha_DIN_Qrange.append(F_f_alpha_DIN_Q5*math.pow(1.4,i-5)) #for 0-4
F_f_alpha_DIN_Q9=F_f_alpha_DIN_Qrange[-1]
for i in range(10,13):
    F_f_alpha_DIN_Qrange.append(F_f_alpha_DIN_Q9*math.pow(1.6,i-9)) #for 10-12
rounding_function(F_f_alpha_DIN_Qrange) #round the numbers
###############################################################################
#temporary
F_f_alpha_DIN_Qrange=[]
F_f_alpha_DIN_Qrange=[0.5,1,1.5,2,3,4.5,6,9,12,16,28,45,71]
###############################################################################
F_f_alpha_left_Q_DIN=0
F_f_alpha_right_Q_DIN=0
F_f_alpha_left_max=0
F_f_alpha_right_max=0
F_f_alpha_left_Q_DIN=findGrade(f_f_alpha_left,F_f_alpha_left_max,F_f_alpha_DIN_Qrange,F_f_alpha_left_Q_DIN)
F_f_alpha_right_Q_DIN=findGrade(f_f_alpha_right,F_f_alpha_right_max,F_f_alpha_DIN_Qrange,F_f_alpha_right_Q_DIN)
#DIN DIN DIN DIN

#==============================================================================
#F_H_alpha_Q5 parameter
F_H_alpha_Q5=0
F_H_alpha_Q5=2*np.sqrt(m_Q)+0.14*np.sqrt(d_Q)+0.5

F_H_alpha_Qrange=[]
for i in range(0,13):
    F_H_alpha_Qrange.append(F_H_alpha_Q5*math.pow(2,0.5*(i-5)))
rounding_function(F_H_alpha_Qrange)

F_H_alpha_left_max=0
F_H_alpha_left_Q=0
F_H_alpha_right_max=0
F_H_alpha_right_Q=0
f_H_alpha_left_Q=findGrade(f_H_alpha_left,F_H_alpha_left_max,F_H_alpha_Qrange,F_H_alpha_left_Q)
f_H_alpha_right_Q=findGrade(f_H_alpha_right,F_H_alpha_right_max,F_H_alpha_Qrange,F_H_alpha_right_Q)
print('Parameter f_H_alpha_left is in ISO grade',f_H_alpha_left_Q)
print('Parameter f_H_alpha_right is in ISO grade',f_H_alpha_right_Q)

#DIN DIN DIN DIN
F_H_alpha_DIN_Q5=0
F_H_alpha_DIN_Q5=2.5+0.25*(m_Q_DIN+3*np.sqrt(m_Q_DIN))

F_H_alpha_DIN_Qrange=[]
for i in range(0,10):  #0-9
    F_H_alpha_DIN_Qrange.append(F_H_alpha_DIN_Q5*math.pow(1.4,i-5)) #for 0-9
F_H_alpha_DIN_Q9=F_H_alpha_DIN_Qrange[-1]
for i in range(10,13):
    F_H_alpha_DIN_Qrange.append(F_H_alpha_DIN_Q9*math.pow(1.6,i-9)) #for 10-12
rounding_function(F_H_alpha_DIN_Qrange) #round the numbers

###############################################################################
#temporary
F_H_alpha_DIN_Qrange=[]
F_H_alpha_DIN_Qrange=[0.5,1,1.5,2,3,4,5,7,10,14,22,36,56]
###############################################################################

F_H_alpha_left_Q_DIN=0
F_H_alpha_right_Q_DIN=0
F_H_alpha_left_Q_DIN=findGrade(f_H_alpha_left,F_H_alpha_left_max,F_H_alpha_DIN_Qrange,F_H_alpha_left_Q_DIN)
F_H_alpha_right_Q_DIN=findGrade(f_H_alpha_right,F_H_alpha_right_max,F_H_alpha_DIN_Qrange,F_H_alpha_right_Q_DIN)
#DIN DIN DIN DIN

#==============================================================================
#F_alpha_Q5 parameter
F_alpha_Q5=0
F_alpha_Q5=3.2*np.sqrt(m_Q)+0.22*np.sqrt(d_Q)+0.7

F_alpha_Qrange=[]
for i in range(0,13):
    F_alpha_Qrange.append(F_alpha_Q5*math.pow(2,0.5*(i-5)))
rounding_function(F_alpha_Qrange)

F_alpha_left_max=0
F_alpha_left_Q=0
F_alpha_right_max=0
F_alpha_right_Q=0
F_alpha_left_Q=findGrade(F_alpha_list_left,F_alpha_left_max,F_alpha_Qrange,F_alpha_left_Q)
F_alpha_right_Q=findGrade(F_alpha_list_right,F_alpha_right_max,F_alpha_Qrange,F_alpha_right_Q)
print('Parameter F_alpha_left is in ISO grade',F_alpha_left_Q)
print('Parameter F_alpha_right is in ISO grade',F_alpha_right_Q)

#DIN DIN DIN DIN
F_alpha_DIN_Q5=0
F_alpha_DIN_Q5=np.sqrt(F_H_alpha_DIN_Q5**2+F_f_alpha_Q5**2)

F_alpha_DIN_Qrange=[]
for i in range(0,10):  #0-9
    F_alpha_DIN_Qrange.append(F_alpha_DIN_Q5*math.pow(1.4,i-5)) #for 0-9
F_alpha_DIN_Q9=F_alpha_DIN_Qrange[-1]
for i in range(10,13):
    F_alpha_DIN_Qrange.append(F_alpha_DIN_Q9*math.pow(1.6,i-9)) #for 10-12
rounding_function(F_alpha_DIN_Qrange) #round the numbers

###############################################################################
#temporary
F_alpha_DIN_Qrange=[]
F_alpha_DIN_Qrange=[0.5,1.5,2,3,4,6,8,12,16,22,36,56,90]
###############################################################################

F_alpha_left_Q_DIN=0
F_alpha_right_Q_DIN=0
F_alpha_left_max=0
F_alpha_right_max=0
F_alpha_left_Q_DIN=findGrade(F_alpha_list_left,F_alpha_left_max,F_alpha_DIN_Qrange,F_alpha_left_Q_DIN)
F_alpha_right_Q_DIN=findGrade(F_alpha_list_right,F_alpha_right_max,F_alpha_DIN_Qrange,F_alpha_right_Q_DIN)
#DIN DIN DIN DIN

#==============================================================================
#F_beta_Q5 parameter
F_beta_Q5=0
F_beta_Q5=0.1*np.sqrt(d_Q)+0.63*np.sqrt(b_Q)+4.2

F_beta_Qrange=[]
for i in range(0,13):
    F_beta_Qrange.append(F_beta_Q5*math.pow(2,0.5*(i-5)))
rounding_function(F_beta_Qrange)

F_beta_left_max=0
F_beta_left_Q=0
F_beta_right_max=0
F_beta_right_Q=0
F_beta_left_Q=findGrade(F_beta_list_left,F_beta_left_max,F_beta_Qrange,F_beta_left_Q)
F_beta_right_Q=findGrade(F_beta_list_right,F_beta_right_max,F_beta_Qrange,F_beta_right_Q)
print('Parameter F_beta_left is in ISO grade',F_beta_left_Q)
print('Parameter F_beta_right is in ISO grade',F_beta_right_Q)

#DIN DIN DIN DIN
F_beta_DIN_Q5=0
F_beta_DIN_Q5=0.8*np.sqrt(b_Q_DIN)+4

F_beta_DIN_Qrange=[]
for i in range(0,7):  #0-6
    F_beta_DIN_Qrange.append(F_beta_DIN_Q5*math.pow(1.25,i-5))
F_beta_DIN_Q6=F_beta_DIN_Qrange[-1]
for i in range(7,9):    #7-8
    F_beta_DIN_Qrange.append(F_beta_DIN_Q6*math.pow(1.4,i-6)) 
F_beta_DIN_Q8=F_beta_DIN_Qrange[-1]
for i in range(9,13): #9-12
    F_beta_DIN_Qrange.append(F_beta_DIN_Q8*math.pow(1.6,i-8))
rounding_function(F_beta_DIN_Qrange) #round the numbers

###############################################################################
#temporary
F_beta_DIN_Qrange=[]
F_beta_DIN_Qrange=[0.5,2.5,3.5,4.5,5.5,7,9,13,18,28,45,71,110]
###############################################################################

F_beta_left_Q_DIN=0
F_beta_right_Q_DIN=0
F_beta_left_max=0
F_beta_right_max=0
F_beta_left_Q_DIN=findGrade(F_beta_list_left,F_beta_left_max,F_beta_DIN_Qrange,F_beta_left_Q_DIN)
F_beta_right_Q_DIN=findGrade(F_beta_list_right,F_beta_right_max,F_beta_DIN_Qrange,F_beta_right_Q_DIN)
#DIN DIN DIN DIN

#==============================================================================
#F_H_beta_Q5 parameter
F_H_beta_Q5=0
F_H_beta_Q5=0.07*np.sqrt(d_Q)+0.45*np.sqrt(b_Q)+3

F_H_beta_Qrange=[]
for i in range(0,13):
    F_H_beta_Qrange.append(F_H_beta_Q5*math.pow(2,0.5*(i-5)))
rounding_function(F_H_beta_Qrange)

F_H_beta_left_max=0
F_H_beta_left_Q=0
F_H_beta_right_max=0
F_H_beta_right_Q=0
f_H_beta_left_Q=findGrade(f_H_beta_left,F_H_beta_left_max,F_H_beta_Qrange,F_H_beta_left_Q)
f_H_beta_right_Q=findGrade(f_H_beta_right,F_H_beta_right_max,F_H_beta_Qrange,F_H_beta_right_Q)
print('Parameter f_H_beta_left is in ISO grade',f_H_beta_left_Q)
print('Parameter f_H_beta_right is in ISO grade',f_H_beta_right_Q)

#DIN DIN DIN DIN
F_H_beta_DIN_Q5=0
F_H_beta_DIN_Q5=4.16*(b_Q_DIN)**0.14

F_H_beta_DIN_Qrange=[]
for i in range(0,7):  #0-6
    F_H_beta_DIN_Qrange.append(F_H_beta_DIN_Q5*math.pow(1.32,i-5))
F_H_beta_DIN_Q6=F_H_beta_DIN_Qrange[-1]
for i in range(7,9): #7-8
    F_H_beta_DIN_Qrange.append(F_H_beta_DIN_Q6*math.pow(1.4,i-6)) 
F_H_beta_DIN_Q8=F_H_beta_DIN_Qrange[-1]
for i in range(9,13): #9-12
    F_H_beta_DIN_Qrange.append(F_H_beta_DIN_Q8*math.pow(1.55,i-8))
rounding_function(F_H_beta_DIN_Qrange) #round the numbers

###############################################################################
#temporary
F_H_beta_DIN_Qrange=[]
F_H_beta_DIN_Qrange=[0.5,2,2.5,3,4,6,8,11,16,25,36,56,90]
###############################################################################

F_H_beta_left_Q_DIN=0
F_H_beta_right_Q_DIN=0
F_H_beta_left_max=0
F_H_beta_right_max=0
F_H_beta_left_Q_DIN=findGrade(f_H_beta_left,F_H_beta_left_max,F_H_beta_DIN_Qrange,F_H_beta_left_Q_DIN)
F_H_beta_right_Q_DIN=findGrade(f_H_beta_right,F_H_beta_right_max,F_H_beta_DIN_Qrange,F_H_beta_right_Q_DIN)
#DIN DIN DIN DIN

#==============================================================================
#F_f_beta_Q5 parameter
F_f_beta_Q5=0
F_f_beta_Q5=0.07*np.sqrt(d_Q)+0.45*np.sqrt(b_Q)+3

F_f_beta_Qrange=[]
for i in range(0,13):
    F_f_beta_Qrange.append(F_f_beta_Q5*math.pow(2,0.5*(i-5)))
rounding_function(F_f_beta_Qrange)

F_f_beta_left_max=0
F_f_beta_left_Q=0
F_f_beta_right_max=0
F_f_beta_right_Q=0
f_f_beta_left_Q=findGrade(f_f_beta_left,F_f_beta_left_max,F_f_beta_Qrange,F_f_beta_left_Q)
f_f_beta_right_Q=findGrade(f_f_beta_right,F_f_beta_right_max,F_f_beta_Qrange,F_f_beta_right_Q)
print('Parameter f_f_beta_left is in ISO grade',f_f_beta_left_Q)
print('Parameter f_f_beta_right is in ISO grade',f_f_beta_right_Q)

#DIN DIN DIN DIN
F_f_beta_DIN_Q5=0
F_f_beta_DIN_Q5=np.sqrt(F_beta_DIN_Q5**2 - F_H_beta_DIN_Q5**2)

F_f_beta_DIN_Qrange=[]
for i in range(0,7):  #0-6
    F_f_beta_DIN_Qrange.append(F_f_beta_DIN_Q5*math.pow(1.3,i-5))
F_f_beta_DIN_Q6=F_f_beta_DIN_Qrange[-1]
for i in range(7,9): #7-8
    F_f_beta_DIN_Qrange.append(F_f_beta_DIN_Q6*math.pow(1.4,i-6)) 
F_f_beta_DIN_Q8=F_f_beta_DIN_Qrange[-1]
for i in range(9,13): #9-12
    F_f_beta_DIN_Qrange.append(F_f_beta_DIN_Q8*math.pow(1.6,i-8))
rounding_function(F_f_beta_DIN_Qrange) #round the numbers

###############################################################################
#temporary
F_f_beta_DIN_Qrange=[]
F_f_beta_DIN_Qrange=[0.5,1.5,2.5,3,3.5,4.5,5.5,7,9,14,25,40,63]
###############################################################################

F_f_beta_left_Q_DIN=0
F_f_beta_right_Q_DIN=0
F_f_beta_left_max=0
F_f_beta_right_max=0
F_f_beta_left_Q_DIN=findGrade(f_f_beta_left,F_f_beta_left_max,F_f_beta_DIN_Qrange,F_f_beta_left_Q_DIN)
F_f_beta_right_Q_DIN=findGrade(f_f_beta_right,F_f_beta_right_max,F_f_beta_DIN_Qrange,F_f_beta_right_Q_DIN)
#DIN DIN DIN DIN

#==============================================================================
#F_r_Q5 parameter
F_r_Q5=0
F_r_Q5=0.24*m_Q+np.sqrt(d_Q)+5.6

F_r_Qrange=[]
for i in range(0,13):
    F_r_Qrange.append(F_r_Q5*math.pow(2,0.5*(i-5)))
rounding_function(F_r_Qrange)

F_r_max=0
F_r_Q=0
F_r_Q=findGrade(F_r,F_r_max,F_r_Qrange,F_r_Q)
print('Parameter F_r is in ISO grade',F_r_Q)

#DIN DIN DIN DIN
F_r_DIN_Q5=0
F_r_DIN_Q5=1.68+2.18*np.sqrt(m_Q_DIN)+(2.3+1.2*np.log10(m_Q_DIN))*(d_Q_DIN)**(1/4)

F_r_DIN_Qrange=[]
for i in range(0,13):  
    F_r_DIN_Qrange.append(F_r_DIN_Q5*math.pow(1.4,i-5)) #for 0-12
rounding_function(F_r_DIN_Qrange) #round the numbers

#temporary
F_r_DIN_Qrange=[]
F_r_DIN_Qrange=[0.5,2.5,3.5,5,7,10,14,20,28,40,56,80,110]
###############################################################################

F_r_DIN_max=0
F_r_DIN_Q=0
F_r_DIN_Q=findGrade(F_r,F_r_DIN_max,F_r_DIN_Qrange,F_r_DIN_Q)
#DIN DIN DIN DIN

#temporary
"New Fi parameter"
F_i_max=0
F_i_Q=0
F_i_Q=findGrade(Fi,F_i_max,F_r_Qrange,F_i_Q)
print('Parameter F_i is in ISO grade',F_i_Q)

#%%
os.chdir(folder)
df = pd.read_csv('odstopki_Cela.asc',sep='\t')
#file = open('odstopki_Cela.asc',"r")    #planar section
os.chdir(owd)
X_cela=[]
Y_cela=[]
Z_cela=[]
dev_cela=[]
for i in range(len(df)):
    if df.iloc[i][3]>0.00001 or df.iloc[i][3]<-0.00001:                #samo ce je vecji od stotinke mikrona
        X_cela.append(df.iloc[i][0])
        Y_cela.append(df.iloc[i][1])
        Z_cela.append(df.iloc[i][2])
        dev_cela.append(df.iloc[i][3])    



circleNindex=[]
for i in range(len(Z_cela)):
    if Z_cela[i] < -2.7 or Z_cela[i] > 2.7:  # odstrani stranske ploskve
        circleNindex.append(i)
c=0
for i in circleNindex:
    X_cela.pop(i-c)
    Y_cela.pop(i-c)
    Z_cela.pop(i-c)
    dev_cela.pop(i-c)
    c=c+1  
                    
r_cela=[]
for i in range(len(X_cela)):
    r_cela.append(math.sqrt(X_cela[i]**2 + Y_cela[i]**2))


radius_b_changed=radius_b*(1+relative_ball_dev/100)


r_cutoff=radius_b_changed+(np.max(r_cela)-radius_b_changed)*val_length_cutoff*2       #define the radius for the cutoff
up_limit=np.max(r_cela)-(np.max(r_cela)-radius_b_changed)*val_length_cutoff*3
circleNindex=[]
for i in range(len(r_cela)):
    if r_cela[i] <r_cutoff or r_cela[i]>up_limit:
        circleNindex.append(i) #a set of indexes for the inner circle
"Remove the inner values"
#Xinner=[]
#Yinner=[]
c=0
for i in circleNindex:
    #Xinner.append(X_cela[i-c])
    #Yinner.append(Y_cela[i-c])
    X_cela.pop(i-c)
    Y_cela.pop(i-c)
    Z_cela.pop(i-c)
    dev_cela.pop(i-c)
    c=c+1    

#%%

"Circular coordinate system"
r_cela=[]
fi_cela=[]
angle=0
for i in range(len(X_cela)):
    r_cela.append(math.sqrt(X_cela[i]**2 + Y_cela[i]**2))
    angle=math.atan2(Y_cela[i],X_cela[i])*(180/np.pi) #angle calculation, return 0<fi<360
    if angle<0:
        angle=angle+360
    fi_cela.append(angle)

"Sort the points by angle fi"
fi_r_sorted_cela=sorted(zip(fi_cela,r_cela,Z_cela,dev_cela))
r_sorted_cela=[r_cela for fi_cela, r_cela,Z_cela,dev_cela in fi_r_sorted_cela]
Z_sorted_cela=[Z_cela for fi_cela, r_cela,Z_cela,dev_cela in fi_r_sorted_cela]
dev_sorted_cela=[dev_cela for fi_cela, r_cela,Z_cela,dev_cela in fi_r_sorted_cela]
fi_sorted_cela=sorted(fi_cela)

"Divide the teeth"
fi_divided_cela=[[] for x in range(N_teeth)]
r_divided_cela=[[] for x in range(N_teeth)]
Z_divided_cela=[[] for x in range(N_teeth)]
dev_divided_cela=[[] for x in range(N_teeth)]
for i in range(len(fi_sorted_cela)):
    if fi_sorted_cela[i]<fi_range/2 or fi_sorted_cela[i]>(360-(fi_range/2)):
        fi_divided_cela[0].append(fi_sorted_cela[i])
        r_divided_cela[0].append(r_sorted_cela[i])
        Z_divided_cela[0].append(Z_sorted_cela[i])
        dev_divided_cela[0].append(dev_sorted_cela[i])
    else:
        for j in range(N_teeth):
            if fi_sorted_cela[i]>fi_range/2+(fi_range*(j-1)) and fi_sorted_cela[i]<(360-fi_range/2-fi_range*(N_teeth-1-j)):
                fi_divided_cela[j].append(fi_sorted_cela[i])
                r_divided_cela[j].append(r_sorted_cela[i])
                Z_divided_cela[j].append(Z_sorted_cela[i])
                dev_divided_cela[j].append(dev_sorted_cela[i])
    

 
fi_divided_val=[[] for x in range(N_teeth)] 
X_divided_cela=[[] for x in range(N_teeth)]
Y_divided_cela=[[] for x in range(N_teeth)]
for i in range(len(fi_divided_cela)):
    for j in range(len(fi_divided_cela[i])):
        fi_divided_val[i].append(fi_divided_cela[i][j]*np.pi/180)                     #turn to radians for plotting                
        X_divided_cela[i].append(r_divided_cela[i][j]*np.cos(fi_divided_val[i][j]))    #need angle in radians
        Y_divided_cela[i].append(r_divided_cela[i][j]*np.sin(fi_divided_val[i][j]))    #need angle in radians
    
 
    
#Lets divide to left and right
r_right_cela=[[] for x in range(N_teeth)]
fi_right_cela=[[] for x in range(N_teeth)]
X_right_cela=[[] for x in range(N_teeth)]
Y_right_cela=[[] for x in range(N_teeth)]
Z_right_cela=[[] for x in range(N_teeth)]
dev_right_cela=[[] for x in range(N_teeth)]
r_left_cela=[[] for x in range(N_teeth)]
fi_left_cela=[[] for x in range(N_teeth)]
X_left_cela=[[] for x in range(N_teeth)]
Y_left_cela=[[] for x in range(N_teeth)]
Z_left_cela=[[] for x in range(N_teeth)]
dev_left_cela=[[] for x in range(N_teeth)]
avg_angle=0

for j in range(len(r_divided_cela[0])): #this is the first tooth
    if fi_divided_cela[0][j]>180:
        r_right_cela[0].append(r_divided_cela[0][j])
        fi_right_cela[0].append(fi_divided_cela[0][j])
        X_right_cela[0].append(X_divided_cela[0][j])
        Y_right_cela[0].append(Y_divided_cela[0][j])
        Z_right_cela[0].append(Z_divided_cela[0][j])
        dev_right_cela[0].append(dev_divided_cela[0][j])
    else:
        r_left_cela[0].append(r_divided_cela[0][j])
        fi_left_cela[0].append(fi_divided_cela[0][j])
        X_left_cela[0].append(X_divided_cela[0][j])
        Y_left_cela[0].append(Y_divided_cela[0][j])
        Z_left_cela[0].append(Z_divided_cela[0][j])
        dev_left_cela[0].append(dev_divided_cela[0][j])
            
for i in range(1,N_teeth): #the remaining teeth
    avg_angle=np.average(fi_divided_cela[i])
    for j in range(len(r_divided_cela[i])):
        if fi_divided_cela[i][j]<avg_angle:
            r_right_cela[i].append(r_divided_cela[i][j])
            fi_right_cela[i].append(fi_divided_cela[i][j])
            X_right_cela[i].append(X_divided_cela[i][j])
            Y_right_cela[i].append(Y_divided_cela[i][j])
            Z_right_cela[i].append(Z_divided_cela[i][j])
            dev_right_cela[i].append(dev_divided_cela[i][j])
        else:
            r_left_cela[i].append(r_divided_cela[i][j])
            fi_left_cela[i].append(fi_divided_cela[i][j])
            X_left_cela[i].append(X_divided_cela[i][j])
            Y_left_cela[i].append(Y_divided_cela[i][j])
            Z_left_cela[i].append(Z_divided_cela[i][j])
            dev_left_cela[i].append(dev_divided_cela[i][j])    
    


# =============================================================================
# For the right flank we need to rotate the teeth over 180 to the first and second quadrant. Our method works there.
# Lets also mirror the left flank to the right, so that we can evaluate it. It works for the right flank.
index_min=0
fi_cut_calc=0
r_left_cela_mirr=[[] for x in range(N_teeth)]
fi_left_cela_mirr=[[] for x in range(N_teeth)]
X_left_cela_mirr=[[] for x in range(N_teeth)]
Y_left_cela_mirr=[[] for x in range(N_teeth)]
#Z_left_cela_mirr=[[] for x in range(N_teeth)]      #tukej je Z in dev lahko enaki kot prej, tako da uporabi tiste vrednosti
#dev_left_cela_mirr=[[] for x in range(N_teeth)]
for i in range(len(X_left_cela)): 
    index_min = np.argmin(r_left_cela[i])
    for j in range(len(r_left_cela[i])):
        r_left_cela_mirr[i].append(r_left_cela[i][j]) #mirror and also Y>0
        fi_cut_calc=fi_left_cela[i][index_min]+(fi_left_cela[i][index_min]-fi_left_cela[i][j])
        if fi_cut_calc>180:
            fi_cut_calc=fi_cut_calc-180
        else:
            fi_cut_calc=fi_cut_calc
        if fi_cut_calc>90:
            fi_cut_calc=fi_cut_calc-90
        else:
            fi_cut_calc=fi_cut_calc
        fi_left_cela_mirr[i].append(fi_cut_calc)
        X_left_cela_mirr[i].append(r_left_cela[i][j]*np.cos(np.radians(fi_cut_calc)))
        Y_left_cela_mirr[i].append(r_left_cela[i][j]*np.sin(np.radians(fi_cut_calc)))
        
for i in range(0,len(X_left_cela)):           
    plt.plot(X_left_cela_mirr[i],Y_left_cela_mirr[i],marker='o', markerfacecolor='black', markersize=1, linestyle=' ')
plt.plot(0, 0,'ro',label='Origin C.S.')
plt.axis('equal')
plt.title('Divided teeth, cutoff, right rotated')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()



#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

#rotate the right flank so its Y>0
fi_cut_calc=0
r_right_cela_mirr=[[] for x in range(N_teeth)]
fi_right_cela_mirr=[[] for x in range(N_teeth)]
X_right_cela_mirr=[[] for x in range(N_teeth)]
Y_right_cela_mirr=[[] for x in range(N_teeth)]
#Z_right_cela_mirr=[[] for x in range(N_teeth)]          #tukej je Z in dev lahko enaki kot prej, tako da uporabi tiste vrednosti
#dev_right_cela_mirr=[[] for x in range(N_teeth)]
for i in range(len(fi_right_cela)):
    for j in range(len(fi_right_cela[i])):
        if fi_right_cela[i][j]>180:
            fi_cut_calc=fi_right_cela[i][j]-180
        else:
            fi_cut_calc=fi_right_cela[i][j]
        if fi_cut_calc>88:                          #this is 90 by default #temporary
            fi_cut_calc=fi_cut_calc-90
        else:
            fi_cut_calc=fi_cut_calc
        r_right_cela_mirr[i].append(r_right_cela[i][j])
        fi_right_cela_mirr[i].append(fi_cut_calc)
        X_right_cela_mirr[i].append(r_right_cela[i][j]*np.cos(np.radians(fi_cut_calc)))
        Y_right_cela_mirr[i].append(r_right_cela[i][j]*np.sin(np.radians(fi_cut_calc)))


for i in range(0,len(X_right_cela)):           
    plt.plot(X_right_cela_mirr[i],Y_right_cela_mirr[i],marker='o', markerfacecolor='black', markersize=1, linestyle=' ')
plt.plot(0, 0,'ro',label='Origin C.S.')
plt.axis('equal')
plt.title('Divided teeth, cutoff, right rotated')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()

#mogoce bi tudi tu lahko dal radius_b_changed ampak ni velike razlike. Pa dati bi potem moral še pod 2D evaluacijo

# RIGHT FLANK
y_T_right_cela=[[] for x in range(N_teeth)]
x_T_right_cela=[[] for x in range(N_teeth)]
v_T_right_cela=0
x_plot_right_cela=[[] for x in range(N_teeth)]
for i in range(len(X_right_cela_mirr)):
    for j in range(len(X_right_cela_mirr[i])):
        y__T=(((radius_b**2)*Y_right_cela_mirr[i][j]+radius_b*X_right_cela_mirr[i][j]*np.sqrt(X_right_cela_mirr[i][j]**2+Y_right_cela_mirr[i][j]**2-(radius_b**2)))/(X_right_cela_mirr[i][j]**2+Y_right_cela_mirr[i][j]**2))
        x__T=((radius_b**2)-Y_right_cela_mirr[i][j]*y__T)/X_right_cela_mirr[i][j]
        v_T=np.arccos(x__T/radius_b)
        x_plot_right_cela[i].append(v_T*radius_b)


# # temporary za testirat roll length
# x_plot_right_cela_nov=[[] for x in range(N_teeth)]
# for i in range(len(X_right_cela_mirr)):
#     for j in range(len(X_right_cela_mirr[i])):
#         x_plot_right_cela_nov[i].append(x_plot_right_cela[i][j]/3)

# x_plot_right_cela=x_plot_right_cela_nov
        
# =============================================================================
# LEFT FLANK
y_T_left_cela=[[] for x in range(N_teeth)]
x_T_left_cela=[[] for x in range(N_teeth)]
v_T_left_cela=0
x_plot_left_cela=[[] for x in range(N_teeth)]
for i in range(len(X_left_cela_mirr)):
    for j in range(len(X_left_cela_mirr[i])):
        y__T=(((radius_b**2)*Y_left_cela_mirr[i][j]+radius_b*X_left_cela_mirr[i][j]*np.sqrt(X_left_cela_mirr[i][j]**2+Y_left_cela_mirr[i][j]**2-(radius_b**2)))/(X_left_cela_mirr[i][j]**2+Y_left_cela_mirr[i][j]**2))
        x__T=((radius_b**2)-Y_left_cela_mirr[i][j]*y__T)/X_left_cela_mirr[i][j]
        v_T=np.arccos(x__T/radius_b)
        x_plot_left_cela[i].append(v_T*radius_b)

# =============================================================================

min_x_plot_right_cela=0
for i in range(len(x_plot_right_cela)):
    min_x_plot_right_cela=np.min(x_plot_right_cela[i])
    for j in range(len(x_plot_right_cela[i])):
        x_plot_right_cela[i][j]=x_plot_right_cela[i][j]-min_x_plot_right_cela
        
min_x_plot_left_cela=0
for i in range(len(x_plot_left_cela)):
    min_x_plot_left_cela=np.min(x_plot_left_cela[i])
    for j in range(len(x_plot_left_cela[i])):
        x_plot_left_cela[i][j]=x_plot_left_cela[i][j]-min_x_plot_left_cela
        
        
    
#%%
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

x = np.asarray(X_divided_cela[10])
y = np.asarray(Y_divided_cela[10])
z = np.asarray(Z_divided_cela[10])
c = np.asarray(dev_divided_cela[10]) 

index_x = 0; index_y = 1; index_z = 2; index_c = 3;
list_name_variables = ['x', 'y', 'z', 'odstopki [mm]'];
name_color_map = 'seismic';
name_color_map_surface = 'Greens';  # Colormap for the 3D surface only.


fig = plt.figure(); 

ax = fig.add_subplot(111, projection='3d');
ax.set_xlabel(list_name_variables[index_x]); ax.set_ylabel(list_name_variables[index_y]);
ax.set_zlabel(list_name_variables[index_z]);
plt.title('%s fun od %s, %s and %s' % (list_name_variables[index_c], list_name_variables[index_x], list_name_variables[index_y], list_name_variables[index_z]) );
ax.view_init(-55, 90)
# In this case, we will have 2 color bars: one for the surface and another for 
# the "scatter plot".
# For example, we can place the second color bar under or to the left of the figure.
choice_pos_colorbar = 2;

#The scatter plot.
img = ax.scatter(x, y, z, c = c, cmap = name_color_map);
cbar = fig.colorbar(img, shrink=0.5, aspect=5); # Default location is at the 'right' of the figure.
cbar.ax.get_yaxis().labelpad = 15; cbar.ax.set_ylabel(list_name_variables[index_c], rotation = 270);

# The 3D surface that serves only to connect the points to help visualize 
# the distances that separates them.
# The "alpha" is used to have some transparency in the surface.
# surf = ax.plot_trisurf(x, y, z, cmap = name_color_map_surface, linewidth = 0.2, alpha = 0.25);

# # The second color bar will be placed at the left of the figure.
# if choice_pos_colorbar == 1: 
#     #I am trying here to have the two color bars with the same size even if it 
#     #is currently set manually.
#     cbaxes = fig.add_axes([1-0.78375-0.1, 0.3025, 0.0393823, 0.385]);  # Case without tigh layout.
#     #cbaxes = fig.add_axes([1-0.844805-0.1, 0.25942, 0.0492187, 0.481161]); # Case with tigh layout.

#     cbar = plt.colorbar(surf, cax = cbaxes, shrink=0.5, aspect=5);
#     cbar.ax.get_yaxis().labelpad = 15; cbar.ax.set_ylabel(list_name_variables[index_z], rotation = 90);

# # The second color bar will be placed under the figure.
# elif choice_pos_colorbar == 2: 
#     cbar = fig.colorbar(surf, shrink=0.75, aspect=20,pad = 0.05, orientation = 'horizontal');
#     cbar.ax.get_yaxis().labelpad = 15; cbar.ax.set_xlabel(list_name_variables[index_z], rotation = 0);
# #end
fig.savefig("3D_1.pdf", bbox_inches='tight')
plt.show();


#plt.savefig("Green_graph.png", dpi = 600, bbox_inches='tight')
#plt.savefig("Red_graph.pdf", bbox_inches='tight')

    

"Zdaj imamo nov plot za raztegnjene vrednosti"   
    
    
fig = plt.figure(); 
ax = plt.axes(projection='3d');
img=ax.scatter(x_plot_right_cela[10], Z_right_cela[10], dev_right_cela[10], c=dev_right_cela[10], cmap='viridis', linewidth=0.5);

#cbar = fig.colorbar(img, shrink=0.5, aspect=5); # Default location is at the 'right' of the figure.
#cbar.ax.get_yaxis().labelpad = 15; cbar.ax.set_ylabel('deviacija', rotation = 270);

ax.set_xlabel('x - rollout length [mm]');
ax.set_zlabel('deviation [mm]');
ax.set_ylabel('gear width [mm]');
#plt.title('')
ax.view_init(10, 290)
    
plt.show();    
    
#%%
"Zaenkrat dela. Lahko že direktno določim Falpha in Fbeta. Max-min"
"Za drugo moram fittat plane in potem lahko na podoben način določim"

import scipy.linalg
#from mpl_toolkits.mplot3d import Axes3D
 
"First for right on all teeth"  

# F_alpha_plane_right=[]
# f_f_alpha_plane_right=[]
# f_H_alpha_plane_right=[]
# F_beta_plane_right=[]
# f_f_beta_plane_right=[]
# f_H_beta_plane_right=[]

F_alphabeta_plane_right=[]
f_f_alphabeta_plane_right=[]
f_H_alpha_plane_right=[]
f_H_beta_plane_right=[]

for i in range(N_teeth):
    # some 3-dim points
    data = np.c_[x_plot_right_cela[i],Z_right_cela[i],dev_right_cela[i]]
    
    # regular grid covering the domain of the data
    mn = np.min(data, axis=0)
    mx = np.max(data, axis=0)
    X,Y = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))
    XX = X.flatten()
    YY = Y.flatten()
    
    # best-fit linear plane
    A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients
        
    # evaluate it on grid
    Z = C[0]*X + C[1]*Y + C[2]
    
    
    # #best-fit linear plane
    # A = np.c_[data[:,0], np.ones(data.shape[0])]
    # C = scipy.linalg.lstsq(A, data[:,2])    # coefficients
    
    # up plane
    maxdevindex=np.argmax(data[:,2])
    C2up=data[maxdevindex,2]-C[0]*data[maxdevindex,0]-C[1]*data[maxdevindex,1]
    Z_up= C[0]*X + C[1]*Y + C2up   
        
    # down plane
    mindevindex=np.argmin(data[:,2])
    C2down=data[mindevindex,2]-C[0]*data[mindevindex,0]-C[1]*data[mindevindex,1]
    Z_down= C[0]*X + C[1]*Y + C2down   
    
    if i == 4:
        # plot points and fitted surface
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
        
        ax.plot_surface(X, Y, Z_up, rstride=1, cstride=1, alpha=0.2)
        ax.plot_surface(X, Y, Z_down, rstride=1, cstride=1, alpha=0.2)
        
        img=ax.scatter(data[:,0], data[:,1], data[:,2], c=data[:,2], s=50, cmap='viridis', linewidth=0.5);
        ax.set_xlabel('x - rollout length [mm]');
        ax.set_zlabel('deviation [mm]');
        ax.set_ylabel('gear width [mm]');
        plt.title('Right')
        ax.view_init(10, 290)
        ax.axis('auto')
        ax.axis('tight')
        plt.show()      
        
        
    F_alphabeta_plane_right.append((np.max(data[:,2])-np.min(data[:,2]))*1000)
    #print('F_alphabeta_plane=',round(F_alphabeta_plane,1),'um')    
    f_f_alphabeta_plane_right.append((C2up-C2down)*1000)
    #print('f_f_alphabeta_plane=',round(f_f_alphabeta_plane,1),'um')    
    f_H_alpha_plane_right.append(C[0]*np.max(data[:,0]) *1000)
    #print('f_H_alpha_plane=',round(f_H_alpha_plane,1),'um') 
    f_H_beta_plane_right.append(C[1]*np.abs(np.min(data[:,1])-np.max(data[:,1]))*1000)
    #print('f_H_beta_plane=',round(f_H_beta_plane,1),'um')     
    
    
    # "Za alpha parametre"
    
    # A = np.vstack([data[:,0], np.ones(data.shape[0])]).T
    # C = scipy.linalg.lstsq(A, data[:,2])[0]
    
    # # evaluate it on grid
    # Z = C[0]*X + C[1]
    
    
    # # up plane
    # maxdevindex=np.argmax(data[:,2])
    # C1up=data[maxdevindex,2]-C[0]*data[maxdevindex,0]
    # Z_up= C[0]*X + C1up   
        
    # # down plane
    # mindevindex=np.argmin(data[:,2])
    # C1down=data[mindevindex,2]-C[0]*data[mindevindex,0]
    # Z_down= C[0]*X + C1down   
    
    
    # F_alpha_plane_right.append((np.max(data[:,2])-np.min(data[:,2]))*1000)
    # #print('F_alpha_plane=',round(F_alpha_plane,1),'um')
    
    
    # f_f_alpha_plane_right.append((C1up-C1down)*1000)
    # #print('f_f_alpha_plane=',round(f_f_alpha_plane,1),'um')    
        
        
    # f_H_alpha_plane_right.append(C[0]*np.max(data[:,0]) *1000)
    # #print('f_H_alpha_plane=',round(f_H_alpha_plane,1),'um') 
    

    # if i == 11:
    #     "Plots in the end - last tooth"
    #     # plot points and fitted surface
    #     fig = plt.figure()
    #     ax = fig.gca(projection='3d')
    #     ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
        
    #     ax.plot_surface(X, Y, Z_up, rstride=1, cstride=1, alpha=0.2)
    #     ax.plot_surface(X, Y, Z_down, rstride=1, cstride=1, alpha=0.2)
        
    #     img=ax.scatter(data[:,0], data[:,1], data[:,2], c=data[:,2], s=50, cmap='viridis', linewidth=0.5);
    #     ax.set_xlabel('x - rollout length [mm]');
    #     ax.set_zlabel('deviation [mm]');
    #     ax.set_ylabel('gear width [mm]');
    #     plt.title('Alpha, right')
    #     ax.view_init(10, 350)
    #     ax.axis('auto')
    #     ax.axis('tight')
    #     plt.show()     
    
    # "=================="
    # "Za beta parametre"
    
    # A = np.vstack([data[:,1], np.ones(data.shape[0])]).T
    # C = scipy.linalg.lstsq(A, data[:,2])[0]
    
    # # evaluate it on grid
    # Z = C[0]*Y + C[1]
    
    
    # # up plane
    # maxdevindex=np.argmax(data[:,2])
    # C1up=data[maxdevindex,2]-C[0]*data[maxdevindex,1]
    # Z_up= C[0]*Y + C1up   
        
    # # down plane
    # mindevindex=np.argmin(data[:,2])
    # C1down=data[mindevindex,2]-C[0]*data[mindevindex,1]
    # Z_down= C[0]*Y + C1down   
    
    # if i == 12:
    #     "Plots in the end - last tooth"
    #     # plot points and fitted surface
    #     fig = plt.figure()
    #     ax = fig.gca(projection='3d')
    #     ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
        
    #     ax.plot_surface(X, Y, Z_up, rstride=1, cstride=1, alpha=0.2)
    #     ax.plot_surface(X, Y, Z_down, rstride=1, cstride=1, alpha=0.2)
        
    #     img=ax.scatter(data[:,0], data[:,1], data[:,2], c=data[:,2], s=50, cmap='viridis', linewidth=0.5);
    #     ax.set_xlabel('x - rollout length [mm]');
    #     ax.set_zlabel('deviation [mm]');
    #     ax.set_ylabel('gear width [mm]');
    #     plt.title('Beta, right')
    #     ax.view_init(10, 290)
    #     ax.axis('auto')
    #     ax.axis('tight')
    #     plt.show()
    
    # F_beta_plane_right.append((np.max(data[:,2])-np.min(data[:,2]))*1000)
    # #print('F_beta_plane=',round(F_beta_plane,1),'um')
    # f_f_beta_plane_right.append((C1up-C1down)*1000)
    # #print('f_f_beta_plane=',round(f_f_beta_plane,1),'um')        
    # f_H_beta_plane_right.append(C[0]*(np.max(data[:,1])-np.min(data[:,1])) *1000)
    # #print('f_H_beta_plane=',round(f_H_beta_plane,1),'um') 

    

# "Plots in the end - last tooth"
# # plot points and fitted surface
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)

# ax.plot_surface(X, Y, Z_up, rstride=1, cstride=1, alpha=0.2)
# ax.plot_surface(X, Y, Z_down, rstride=1, cstride=1, alpha=0.2)

# img=ax.scatter(data[:,0], data[:,1], data[:,2], c=data[:,2], s=50, cmap='viridis', linewidth=0.5);
# ax.set_xlabel('x - rollout length [mm]');
# ax.set_zlabel('deviation [mm]');
# ax.set_ylabel('gear width [mm]');
# plt.title('Alpha, right')
# ax.view_init(10, 290)
# ax.axis('auto')
# ax.axis('tight')
# plt.show()    




# # plot points and fitted surface
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)

# ax.plot_surface(X, Y, Z_up, rstride=1, cstride=1, alpha=0.2)
# ax.plot_surface(X, Y, Z_down, rstride=1, cstride=1, alpha=0.2)

# img=ax.scatter(data[:,0], data[:,1], data[:,2], c=data[:,2], s=50, cmap='viridis', linewidth=0.5);
# ax.set_xlabel('x - rollout length [mm]');
# ax.set_zlabel('deviation [mm]');
# ax.set_ylabel('gear width [mm]');
# plt.title('Beta,right')
# ax.view_init(10, 290)
# ax.axis('auto')
# ax.axis('tight')
# plt.show()      
    

"Determine worst grades"

maxdevindex=np.argmax(np.abs(F_alphabeta_plane_right))
F_alphabeta_plane_right_value=F_alphabeta_plane_right[maxdevindex]
F_alphabeta_plane_right_average=np.average(F_alphabeta_plane_right)
F_alphabeta_plane_right_std=np.std(F_alphabeta_plane_right)
F_alphabeta_right_max=0
F_alphabeta_plane_right_Q=0
F_alphabeta_plane_right_Q=findGrade(F_alphabeta_plane_right_value,F_alphabeta_right_max,F_alpha_Qrange,F_alphabeta_plane_right_Q)
print('F_alphabeta_plane_right=',round(F_alphabeta_plane_right_value,1),'um; on tooth',maxdevindex,'; Q=',F_alphabeta_plane_right_Q)
    
maxdevindex=np.argmax(np.abs(f_f_alphabeta_plane_right))
f_f_alphabeta_plane_right_value=f_f_alphabeta_plane_right[maxdevindex]
f_f_alphabeta_plane_right_average=np.average(f_f_alphabeta_plane_right)
f_f_alphabeta_plane_right_std=np.std(f_f_alphabeta_plane_right)
f_f_alphabeta_right_max=0
f_f_alphabeta_plane_right_Q=0
f_f_alphabeta_plane_right_Q=findGrade(f_f_alphabeta_plane_right_value,f_f_alphabeta_right_max,F_f_alpha_Qrange,f_f_alphabeta_plane_right_Q)
print('f_f_alphabeta_plane_right=',round(f_f_alphabeta_plane_right_value,1),'um; on tooth',maxdevindex,'; Q=',f_f_alphabeta_plane_right_Q)

maxdevindex=np.argmax(np.abs(f_H_alpha_plane_right))
f_H_alpha_plane_right_value=f_H_alpha_plane_right[maxdevindex]
f_H_alpha_plane_right_average=np.average(f_H_alpha_plane_right)
f_H_alpha_plane_right_std=np.std(f_H_alpha_plane_right)
f_H_alpha_right_max=0
f_H_alpha_plane_right_Q=0
f_H_alpha_plane_right_Q=findGrade(f_H_alpha_plane_right_value,f_H_alpha_right_max,F_H_alpha_Qrange,f_H_alpha_plane_right_Q)
print('f_H_alpha_plane_right=',round(f_H_alpha_plane_right_value,1),'um; on tooth',maxdevindex,'; Q=',f_H_alpha_plane_right_Q)

# 'beta'
# maxdevindex=np.argmax(np.abs(F_beta_plane_right))
# F_beta_plane_right_value=F_beta_plane_right[maxdevindex]
# F_beta_plane_right_average=np.average(F_beta_plane_right)
# F_beta_plane_right_std=np.std(F_beta_plane_right)
# F_beta_right_max=0
# F_beta_plane_right_Q=0
# F_beta_plane_right_Q=findGrade(F_beta_plane_right_value,F_beta_right_max,F_beta_Qrange,F_beta_plane_right_Q)
# print('F_beta_plane_right=',round(F_beta_plane_right_value,1),'um; on tooth',maxdevindex,'; Q=',F_beta_plane_right_Q)
    
# maxdevindex=np.argmax(np.abs(f_f_beta_plane_right))
# f_f_beta_plane_right_value=f_f_beta_plane_right[maxdevindex]
# f_f_beta_plane_right_average=np.average(f_f_beta_plane_right)
# f_f_beta_plane_right_std=np.std(f_f_beta_plane_right)
# f_f_beta_right_max=0
# f_f_beta_plane_right_Q=0
# f_f_beta_plane_right_Q=findGrade(f_f_beta_plane_right_value,f_f_beta_right_max,F_f_beta_Qrange,f_f_beta_plane_right_Q)
# print('f_f_beta_plane_right=',round(f_f_beta_plane_right_value,1),'um; on tooth',maxdevindex,'; Q=',f_f_beta_plane_right_Q)

maxdevindex=np.argmax(np.abs(f_H_beta_plane_right))
f_H_beta_plane_right_value=f_H_beta_plane_right[maxdevindex]
f_H_beta_plane_right_average=np.average(f_H_beta_plane_right)
f_H_beta_plane_right_std=np.std(f_H_beta_plane_right)
f_H_beta_right_max=0
f_H_beta_plane_right_Q=0
f_H_beta_plane_right_Q=findGrade(f_H_beta_plane_right_value,f_H_beta_right_max,F_H_beta_Qrange,f_H_beta_plane_right_Q)
print('f_H_beta_plane_right=',round(f_H_beta_plane_right_value,1),'um; on tooth',maxdevindex,'; Q=',f_H_beta_plane_right_Q)

#%%
"==========================="
"LEFT on all teeth"  

F_alphabeta_plane_left=[]
f_f_alphabeta_plane_left=[]
f_H_alpha_plane_left=[]
f_H_beta_plane_left=[]

for i in range(N_teeth):
    # some 3-dim points
    data = np.c_[x_plot_left_cela[i],Z_left_cela[i],dev_left_cela[i]]
    
    # regular grid covering the domain of the data
    mn = np.min(data, axis=0)
    mx = np.max(data, axis=0)
    X,Y = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))
    XX = X.flatten()
    YY = Y.flatten()
    
    # best-fit linear plane
    A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients
        
    # evaluate it on grid
    Z = C[0]*X + C[1]*Y + C[2]
    
    
    # up plane
    maxdevindex=np.argmax(data[:,2])
    C2up=data[maxdevindex,2]-C[0]*data[maxdevindex,0]-C[1]*data[maxdevindex,1]
    Z_up= C[0]*X + C[1]*Y + C2up   
        
    # down plane
    mindevindex=np.argmin(data[:,2])
    C2down=data[mindevindex,2]-C[0]*data[mindevindex,0]-C[1]*data[mindevindex,1]
    Z_down= C[0]*X + C[1]*Y + C2down   
    
    if i==0:
        # plot points and fitted surface
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
        
        ax.plot_surface(X, Y, Z_up, rstride=1, cstride=1, alpha=0.2)
        ax.plot_surface(X, Y, Z_down, rstride=1, cstride=1, alpha=0.2)
        
        img=ax.scatter(data[:,0], data[:,1], data[:,2], c=data[:,2], s=50, cmap='viridis', linewidth=0.5);
        ax.set_xlabel('x - rollout length [mm]');
        ax.set_zlabel('deviation [mm]');
        ax.set_ylabel('gear width [mm]');
        plt.title('Left')
        ax.view_init(10, 290)
        ax.axis('auto')
        ax.axis('tight')
        fig.savefig("3D_2.pdf", bbox_inches='tight')
        plt.show()   
        
        
        
    F_alphabeta_plane_left.append((np.max(data[:,2])-np.min(data[:,2]))*1000)
    #print('F_alphabeta_plane=',round(F_alphabeta_plane,1),'um')    
    f_f_alphabeta_plane_left.append((C2up-C2down)*1000)
    #print('f_f_alphabeta_plane=',round(f_f_alphabeta_plane,1),'um')    
    f_H_alpha_plane_left.append(C[0]*np.max(data[:,0]) *1000)
    #print('f_H_alpha_plane=',round(f_H_alpha_plane,1),'um') 
    f_H_beta_plane_left.append(C[1]*np.abs(np.min(data[:,1])-np.max(data[:,1]))*1000)
    #print('f_H_beta_plane=',round(f_H_beta_plane,1),'um')     
    

"Determine worst grades"

maxdevindex=np.argmax(np.abs(F_alphabeta_plane_left))
F_alphabeta_plane_left_value=F_alphabeta_plane_left[maxdevindex]
F_alphabeta_plane_left_average=np.average(F_alphabeta_plane_left)
F_alphabeta_plane_left_std=np.std(F_alphabeta_plane_left)
F_alphabeta_left_max=0
F_alphabeta_plane_left_Q=0
F_alphabeta_plane_left_Q=findGrade(F_alphabeta_plane_left_value,F_alphabeta_left_max,F_alpha_Qrange,F_alphabeta_plane_left_Q)
print('F_alphabeta_plane_left=',round(F_alphabeta_plane_left_value,1),'um; on tooth',maxdevindex,'; Q=',F_alphabeta_plane_left_Q)
    
maxdevindex=np.argmax(np.abs(f_f_alphabeta_plane_left))
f_f_alphabeta_plane_left_value=f_f_alphabeta_plane_left[maxdevindex]
f_f_alphabeta_plane_left_average=np.average(f_f_alphabeta_plane_left)
f_f_alphabeta_plane_left_std=np.std(f_f_alphabeta_plane_left)
f_f_alphabeta_left_max=0
f_f_alphabeta_plane_left_Q=0
f_f_alphabeta_plane_left_Q=findGrade(f_f_alphabeta_plane_left_value,f_f_alphabeta_left_max,F_f_alpha_Qrange,f_f_alphabeta_plane_left_Q)
print('f_f_alphabeta_plane_left=',round(f_f_alphabeta_plane_left_value,1),'um; on tooth',maxdevindex,'; Q=',f_f_alphabeta_plane_left_Q)

maxdevindex=np.argmax(np.abs(f_H_alpha_plane_left))
f_H_alpha_plane_left_value=f_H_alpha_plane_left[maxdevindex]
f_H_alpha_plane_left_average=np.average(f_H_alpha_plane_left)
f_H_alpha_plane_left_std=np.std(f_H_alpha_plane_left)
f_H_alpha_left_max=0
f_H_alpha_plane_left_Q=0
f_H_alpha_plane_left_Q=findGrade(f_H_alpha_plane_left_value,f_H_alpha_left_max,F_H_alpha_Qrange,f_H_alpha_plane_left_Q)
print('f_H_alpha_plane_left=',round(f_H_alpha_plane_left_value,1),'um; on tooth',maxdevindex,'; Q=',f_H_alpha_plane_left_Q)


maxdevindex=np.argmax(np.abs(f_H_beta_plane_left))
f_H_beta_plane_left_value=f_H_beta_plane_left[maxdevindex]
f_H_beta_plane_left_average=np.average(f_H_beta_plane_left)
f_H_beta_plane_left_std=np.std(f_H_beta_plane_left)
f_H_beta_left_max=0
f_H_beta_plane_left_Q=0
f_H_beta_plane_left_Q=findGrade(f_H_beta_plane_left_value,f_H_beta_left_max,F_H_beta_Qrange,f_H_beta_plane_left_Q)
print('f_H_beta_plane_left=',round(f_H_beta_plane_left_value,1),'um; on tooth',maxdevindex,'; Q=',f_H_beta_plane_left_Q)


#%%
# os.chdir(folder)
# df = pd.read_csv('odstopki_Cela.asc',sep='\t')
# #file = open('odstopki_Cela.asc',"r")    #planar section
# os.chdir(owd)
# X_cela=[]
# Y_cela=[]
# Z_cela=[]
# dev_cela=[]
# for i in range(len(df)):
#     #if df.iloc[i][3]>0.00001:                #samo ce je vecji od stotinke mikrona
#         X_cela.append(df.iloc[i][0])
#         Y_cela.append(df.iloc[i][1])
#         Z_cela.append(df.iloc[i][2])
#         dev_cela.append(df.iloc[i][3])   

# x = np.asarray(X_cela)
# y = np.asarray(Y_cela)
# z = np.asarray(Z_cela)
# c = np.asarray(dev_cela) 
#%%
# index_x = 0; index_y = 1; index_z = 2; index_c = 3;
# list_name_variables = ['x', 'y', 'z', 'odstopki [mm]'];
# name_color_map = 'seismic';
# name_color_map_surface = 'Greens';  # Colormap for the 3D surface only.
# fig = plt.figure(); 

# ax = fig.add_subplot(111, projection='3d');
# ax.set_xlabel(list_name_variables[index_x]); ax.set_ylabel(list_name_variables[index_y]);
# ax.set_zlabel(list_name_variables[index_z]);
# plt.title('%s fun od %s, %s and %s' % (list_name_variables[index_c], list_name_variables[index_x], list_name_variables[index_y], list_name_variables[index_z]) );
# ax.view_init(-20, 90)

# choice_pos_colorbar = 2;

# #The scatter plot.
# img = ax.scatter(x, y, z, c = c, cmap = name_color_map);

# cbar = fig.colorbar(img, shrink=0.5, aspect=5); # Default location is at the 'right' of the figure.
# plt.clim(-0.08,0.08);
# cbar.ax.get_yaxis().labelpad = 15; cbar.ax.set_ylabel(list_name_variables[index_c], rotation = 270);

# plt.show();
    
    
    

#%%
# Save to an excel file
#"""
tooth_space=[i for i in range(1,N_teeth+1)]
tooth_space.append('f_pt_max')
tooth_space.append('f_pt_ISO Quality grade Q')
tooth_space.append('f_pt_DIN Quality grade Q')
F_p_string='F_p_'+str(k_eval_teeth)
tooth_space.append(F_p_string)
F_p_Q_string='F_p_'+str(k_eval_teeth)+'_ISO Quality grade Q'
tooth_space.append(F_p_Q_string)
tooth_space.append('F_p')
tooth_space.append('F_p_ISO Quality grade Q')
tooth_space.append('F_p_DIN Quality grade Q')
# =============================================================================
f_pt_l_display=[]

for i in range(len(f_pt_left)):
    f_pt_l_display.append(f_pt_left[i])
f_pt_l_display.append(f_pt_max_left)
f_pt_l_display.append(f_pt_left_Q)
f_pt_l_display.append(f_pt_left_Q_DIN)
f_pt_l_display.append(Fpk_max_diff_left)
f_pt_l_display.append(F_pk_left_Q)
f_pt_l_display.append(F_p_left)
f_pt_l_display.append(F_p_left_Q)
f_pt_l_display.append(F_p_left_Q_DIN)

f_pt_r_display=[]
for i in range(len(f_pt_right)):
    f_pt_r_display.append(f_pt_right[i])
f_pt_r_display.append(f_pt_max_right)
f_pt_r_display.append(f_pt_right_Q)
f_pt_r_display.append(f_pt_right_Q_DIN)
f_pt_r_display.append(Fpk_max_diff_right)
f_pt_r_display.append(F_pk_right_Q)
f_pt_r_display.append(F_p_right)
f_pt_r_display.append(F_p_right_Q)
f_pt_r_display.append(F_p_right_Q_DIN)

involute_tooth_column=[i for i in range(0,N_teeth)]
involute_tooth_column.append('ISO Quality grade Q')
involute_tooth_column.append('DIN Quality grade Q')

F_beta_list_right.append(F_beta_right_Q)
F_beta_list_right.append(F_beta_right_Q_DIN)
F_beta_list_left.append(F_beta_left_Q)
F_beta_list_left.append(F_beta_left_Q_DIN)
f_f_beta_right.append(f_f_beta_right_Q)
f_f_beta_right.append(F_f_beta_right_Q_DIN)
f_f_beta_left.append(f_f_beta_left_Q)
f_f_beta_left.append(F_f_beta_left_Q_DIN)
f_H_beta_right.append(f_H_beta_right_Q)
f_H_beta_right.append(F_H_beta_right_Q_DIN)
f_H_beta_left.append(f_H_beta_left_Q)
f_H_beta_left.append(F_H_beta_left_Q_DIN)
# =============================================================================
F_alpha_list_right.append(F_alpha_right_Q)
F_alpha_list_right.append(F_alpha_right_Q_DIN)
F_alpha_list_left.append(F_alpha_left_Q)
F_alpha_list_left.append(F_alpha_left_Q_DIN)
f_f_alpha_right.append(f_f_alpha_right_Q)
f_f_alpha_right.append(F_f_alpha_right_Q_DIN)
f_f_alpha_left.append(f_f_alpha_left_Q)
f_f_alpha_left.append(F_f_alpha_left_Q_DIN)
f_H_alpha_right.append(f_H_alpha_right_Q)
f_H_alpha_right.append(F_H_alpha_right_Q_DIN)
f_H_alpha_left.append(f_H_alpha_left_Q)
f_H_alpha_left.append(F_H_alpha_left_Q_DIN)
# =============================================================================
# =============================================================================
# measure_with_tol1=str(measure1)+' tol '+'+'+str(tol1upper)+str(tol1lower)
# measure_with_tol2=str(measure2)+' tol '+'+'+str(tol2upper)+str(tol2lower)
# =============================================================================

Data = {'General Parameters': ['N_teeth','module'],
        'General Data': [N_teeth,module],  
        'Tooth space&para': tooth_space,
        'f_pt_left&f_pt_max&F_p [um]': f_pt_l_display,
        'f_pt_right&f_pt_max&F_p [um]': f_pt_r_display,
        'Tooth': involute_tooth_column,
        'F_beta_right [um]': F_beta_list_right,
        'F_beta_left [um]': F_beta_list_left,
        'f_{f_beta_right} [um]': f_f_beta_right,
        'f_{f_beta_left} [um]': f_f_beta_left,
        'f_{H_beta_right} [um]':f_H_beta_right,
        'f_{H_beta_left} [um]':f_H_beta_left,
        'F_alpha_right [um]': F_alpha_list_right,
        'F_alpha_left [um]': F_alpha_list_left,
        'f_{f_alpha_right} [um]': f_f_alpha_right,
        'f_{f_alpha_left} [um]': f_f_alpha_left,
        'f_{H_alpha_right} [um]':f_H_alpha_right,
        'f_{H_alpha_left} [um]':f_H_alpha_left,
        'Runout Parameters':['probing_ball_radius [mm]', 'F_r [um]','F_r_ISO Quality grade Q','F_r_DIN Quality grade Q','Average ball distance','Ideal ball position','Relative ball deviation','Standard deviation ball','Fi','Fi ISO Q'],
        'Runout Data':[radius_circle,F_r,F_r_Q,F_r_DIN_Q,average_ball_distance,ideal_ball_pos,relative_ball_dev,st_dev_ball,Fi,F_i_Q],
        'Evaluated dimensions' :['Gear hole within tolerances?','Prescribed measure [mm]','Measured value [mm]','Gear width within tolerances?','Prescribed measure [mm]','Measured value [mm]'],
        '3D parameters':['F_alphabeta_plane_right [um]', 'f_f_alphabeta_plane_right [um]','f_H_alpha_plane_right [um]','f_H_beta_plane_right [um]',
                         'F_alphabeta_plane_left [um]','f_f_alphabeta_plane_left [um]','f_H_alpha_plane_left [um]','f_H_beta_plane_left [um]'],
        'parameter values':[F_alphabeta_plane_right_value, f_f_alphabeta_plane_right_value,f_H_alpha_plane_right_value,f_H_beta_plane_right_value ,
                         F_alphabeta_plane_left_value ,f_f_alphabeta_plane_left_value ,f_H_alpha_plane_left_value ,f_H_beta_plane_left_value ],
        'ISO grade':[F_alphabeta_plane_right_Q, f_f_alphabeta_plane_right_Q,f_H_alpha_plane_right_Q,f_H_beta_plane_right_Q ,
                         F_alphabeta_plane_left_Q ,f_f_alphabeta_plane_left_Q ,f_H_alpha_plane_left_Q ,f_H_beta_plane_left_Q ],
        'average':[F_alphabeta_plane_right_average, f_f_alphabeta_plane_right_average,f_H_alpha_plane_right_average,f_H_beta_plane_right_average ,
                         F_alphabeta_plane_left_average ,f_f_alphabeta_plane_left_average ,f_H_alpha_plane_left_average ,f_H_beta_plane_left_average ],
        'stdev':[F_alphabeta_plane_right_std, f_f_alphabeta_plane_right_std,f_H_alpha_plane_right_std,f_H_beta_plane_right_std ,
                         F_alphabeta_plane_left_std ,f_f_alphabeta_plane_left_std ,f_H_alpha_plane_left_std ,f_H_beta_plane_left_std ]
# =============================================================================
#         'MBD y/n' :[dinbounds,measure_with_tol1,d_inner,winbounds,measure_with_tol2,gear_width]
# =============================================================================
        }

df1 = DataFrame(Data, columns= ['General Parameters','General Data'])

df2 = DataFrame(Data, columns= ['Tooth space&para','f_pt_left&f_pt_max&F_p [um]',
                                'f_pt_right&f_pt_max&F_p [um]'])

df3 = DataFrame(Data, columns= ['Tooth','F_alpha_right [um]','F_alpha_left [um]',
                                'f_{f_alpha_right} [um]','f_{f_alpha_left} [um]',
                                'f_{H_alpha_right} [um]','f_{H_alpha_left} [um]'])

df4 = DataFrame(Data, columns= ['Tooth','F_beta_right [um]','F_beta_left [um]',
                                'f_{f_beta_right} [um]','f_{f_beta_left} [um]',
                                'f_{H_beta_right} [um]','f_{H_beta_left} [um]'])

df5 = DataFrame(Data, columns= ['Runout Parameters','Runout Data'])

df6 = DataFrame(Data, columns= ['Evaluated dimensions','MBD y/n'])

df7 = DataFrame(Data, columns= ['3D parameters','parameter values','ISO grade','average','stdev'])

os.chdir(gearwd) # Writes the file to the working directory
#export_excel = df.to_excel ('report1.xlsx', index = None, header=True)

with pd.ExcelWriter('report.xlsx') as writer:
    df1.to_excel(writer, sheet_name='General',index = None, header=True)
    worksheet = writer.sheets['General']
    worksheet.set_column('A:B', 20)
    df2.to_excel(writer, sheet_name='Pitch control',index = None, header=True)
    worksheet = writer.sheets['Pitch control']
    worksheet.set_column('A:C', 28)
    df3.to_excel(writer, sheet_name='Profile deviation',index = None, header=True)
    worksheet = writer.sheets['Profile deviation']
    worksheet.set_column('A:G', 20)
    df4.to_excel(writer, sheet_name='Lead profile deviation',index = None, header=True)
    worksheet = writer.sheets['Lead profile deviation']
    worksheet.set_column('A:G', 19)
    df5.to_excel(writer, sheet_name='Runout evaluation',index = None, header=True)
    worksheet = writer.sheets['Runout evaluation']
    worksheet.set_column('A:B', 25)
    df6.to_excel(writer, sheet_name='MBD',index = None, header=True)
    worksheet = writer.sheets['MBD']
    worksheet.set_column('A:B', 27)
    df7.to_excel(writer, sheet_name='3D parameters',index = None, header=True)
    worksheet = writer.sheets['3D parameters']
    worksheet.set_column('A:E', 25)
     
#%%
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # ADD PLOT    
n = 0
wb = openpyxl.load_workbook('report.xlsx')
sheets = wb.sheetnames

# General
ws = wb[sheets[0]]
img = openpyxl.drawing.image.Image('General.png')
img.anchor = 'D2'
ws.add_image(img)

# Division evaluation
ws = wb[sheets[1]]
img = openpyxl.drawing.image.Image('f_pt_left.png')
img.anchor = 'E2'
ws.add_image(img)

img = openpyxl.drawing.image.Image('F_pk_left.png')
img.anchor = 'E33'
ws.add_image(img)

img = openpyxl.drawing.image.Image('f_pt_right.png')
img.anchor = 'T2'
ws.add_image(img)

img = openpyxl.drawing.image.Image('F_pk_right.png')
img.anchor = 'T33'
ws.add_image(img)

# Involute evaluation
ws = wb[sheets[2]]
img = openpyxl.drawing.image.Image('F_alpha.png')
img.anchor = 'I2'
ws.add_image(img)

img = openpyxl.drawing.image.Image('f_f_alpha.png')
img.anchor = 'I33'
ws.add_image(img)

img = openpyxl.drawing.image.Image('f_H_alpha.png')
img.anchor = 'X2'
ws.add_image(img)

# Flank side evaluation
ws = wb[sheets[3]]
img = openpyxl.drawing.image.Image('F_beta.png')
img.anchor = 'I2'
ws.add_image(img)

img = openpyxl.drawing.image.Image('f_f_beta.png')
img.anchor = 'I33'
ws.add_image(img)

img = openpyxl.drawing.image.Image('f_H_beta.png')
img.anchor = 'X2'
ws.add_image(img)

# Runout evaluation
ws = wb[sheets[4]]
img = openpyxl.drawing.image.Image('F_r.png')
img.anchor = 'D2'
ws.add_image(img)

wb.save('report.xlsx')
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
os.chdir(owd)
#"""
    
