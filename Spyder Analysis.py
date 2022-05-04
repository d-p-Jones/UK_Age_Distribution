# -*- coding: utf-8 -*-
"""
Spyder Editor

UK Age distribution analysis.
"""

import importlib.util
import sys

def package_check(name):

    if name in sys.modules:
        print(f"{name!r} already in sys.modules")
    elif (spec := importlib.util.find_spec(name)) is not None:
        # If you choose to perform the actual import ...
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        print(f"{name!r} has been imported")
    else:
        print(f"can't find the {name!r} module")






import pandas as pd
import glob 
import matplotlib.pyplot as plt
import os
import geopandas as gpd
import numpy as np
import matplotlib
import matplotlib.animation as animation
#import seaborn as sns
import math






""" Load the data from 2001-2019. """
""" It's in CSV files with the first row being the header information."""
""" get the right drive and create a list of filename"""
""" For this one I already have a csv of all info combined"""
os.chdir('G:/Documents/Data analysis/Age UK dist/Age Dist Spider/agedist')
path = 'Data'
filenames = glob.glob(path + "/*.csv")
dfs = []




""" loop through the filenames reading csv files, formatting correctly, and making wide data into useful long data"""


"""\~~~~~~~~~~~~~~~~~~~~~~~/"""
""" General Extract and Transform   """
"""\~~~~~~~~~~~~~~~~~~~~~~~/"""
ProperData = 'Data/MYEB1_detailed_population_estimates_series_UK_(2019).csv'
big_frame = pd.read_csv(ProperData, encoding = "ISO-8859-1") #get the wide data from csv
list2 = big_frame.columns[:5]                  #The stuff we want to keep as is
big_frame = big_frame.melt(list2, var_name='cols') #wide to long
big_frame[['name1','year']]=big_frame['cols'].str.split('_',expand=True) #split combined columns into two
big_frame = big_frame.drop(columns=['cols','name1'])    #remove the unneccesary cols
big_frame=big_frame.rename(columns={'ladcode19':'geogcode','laname19':'geogname'}) #give it some more sensible names
big_frame=big_frame[['country','geogcode','geogname','sex','year','age','value']]   #drop some more data
little_frame = big_frame[['sex','year','age','value']] #select the bits you want
totals = little_frame.pivot_table(index = ['sex','year','age'], values=['value'],aggfunc='sum') #pivot it



"""\~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~/"""
"""     Total Population Change over time   """
"""\~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~/"""


### data manipulation

#pop_time = little_frame
pop_time = little_frame.pivot_table(index = ['year'], values=['value'],aggfunc='sum') #pivot it
pop_time = pop_time.reset_index()
pop_time['year']=pop_time['year'].astype(float)



# For animations need an interpolated data set
def interp_df(df,step): #for nice smooth transistions in the animation data between points need to be interpolated
    newdf1 = pd.DataFrame(np.arange(min(df.iloc[:, 0]),max(df.iloc[:, 0])+step, step).tolist())
    newdf1[0]=newdf1[0].astype(float)
    newdf1[0]=round(newdf1[0],8)
    #av_year['year']=round(av_year['year'],8)
    newthing = pd.merge(newdf1,df,how='left', left_on=newdf1.columns[0], right_on=df.columns[0])
    newthing = newthing.interpolate(method='linear')
    newthing=newthing.drop(columns=newthing.columns[1])
    return newthing

interped_pop_time = interp_df(pop_time,0.1)
interped_pop_time.columns=['year','population']
#av_year['mean_age']=av_year['mean_age'].astype(float)


### Plot and animate


fig, ax = plt.subplots(dpi=300) #generate figure
plt.style.use('seaborn-whitegrid') #style figure
ax.set_xlim(np.min(pop_time['year']),np.max(pop_time['year'])) #setup axes
ax.set_ylim(np.min(interped_pop_time['population']),np.max(interped_pop_time['population'])) #setup axes


#ax.set_xticklabels(['{:}'.format(int(x)) for x in ax.get_xticks().tolist()])
#ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])


ticks =  ax.get_xticks() #list of tick marks on x axis
ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
ax.set_xticklabels(['{:}'.format(int(abs(tick))) for tick in ticks]) #format tick marks

ticks =  ax.get_yticks()
ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
ax.set_yticklabels(['{:,}'.format(int(abs(tick))) for tick in ticks])


ax.set_xlabel('Year')
ax.set_ylabel('Number of People')
ax.set_title('Total Population of the UK 2001-2019')
line,=ax.plot(0,0,color='darkgray',linewidth=4)
line2,=ax.plot(0,0,marker='o',ls="",markersize=6,color='black')


def animation_frame(i):
    #print(i) ### constantly update console to check all running smoothly
     
    j=math.floor((i/10)) #a dot for every 10 iterations
    data = interped_pop_time.iloc[:int(i+1)] #select data range
    line.set_xdata(data['year'])
    line.set_ydata(data['population'])
    data2 = pop_time.iloc[:int(j+1)]
    line2.set_xdata(data2['year'].astype(float))
    line2.set_ydata(data2['value'].astype(float))
    
    return line,line2


writer = animation.writers['ffmpeg'](fps=40, metadata=dict(artist='Me'), bitrate=1800)
animation1 =matplotlib.animation.FuncAnimation(fig, func=animation_frame,frames=181,blit=True,repeat=False)
animation1.save('pop_growth.mov', writer=writer)
fig.figure.savefig('pop_growth.svg')
fig.figure.savefig('pop_growth.png')





"""\~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~/"""
"""     Population pyramid for 2019    """
"""\~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~/"""

####horizontal bar graph males on one side female on the other
#### data manipulation
little_frame = big_frame[['sex','year','age','value']] #select the bits you want
totals = little_frame.pivot_table(index = ['sex','year','age'], values=['value'],aggfunc='sum') #pivot it
totals=totals.reset_index()
sex_1=totals[totals['sex']==1]
sex_1=sex_1[sex_1['year']=='2019'] #only need 2019 for this
sex_1=sex_1.reset_index()
sex_1['age_labels']=sex_1['age'].astype(str)
sex_2=totals[totals['sex']==2]
sex_2=sex_2[totals['year']=='2019']
sex_2['value']=(0-sex_2['value'])
sex_2=sex_2.reset_index()
sex_2['age_labels']=sex_2['age'].astype(str)

#



#set up the chart display
fig, ax = plt.subplots(dpi=300,figsize=(4,5))

plt.style.use('seaborn-whitegrid')
ax.set_xlim(-500000,500000)
ax.set_ylim(0,90)
ticks =  ax.get_xticks()
ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
ax.set_xticklabels(['{:,}'.format(int(abs(tick))) for tick in ticks])

ax.set_xlabel('Count of People')
ax.set_ylabel('Age')
ax.set_title('UK Population Pyramid 2019')
hfont = {'fontname':'Arial'}
box_style = dict(facecolor='white', edgecolor='white', boxstyle='round')
im = plt.imread('g26215.png')
from matplotlib.offsetbox import  OffsetImage, AnnotationBbox
ax.add_artist( #adding female icon to axes
        AnnotationBbox(
            OffsetImage(im,zoom=0.2)
            , (-400000, 80)
            , frameon=False
        ) 
)


im1 = plt.imread('g26211.png')
ax.add_artist( #adding male icon to axes
        AnnotationBbox(
            OffsetImage(im1,zoom=0.2)
            , (400000, 80)
            , frameon=False
        ) 
)


#animate data additions
#writergif = animation.PillowWriter(fps=40) 
#Writer = animation.writers['ffmpeg']
writer = animation.writers['ffmpeg'](fps=40, metadata=dict(artist='Me'), bitrate=1800)


def animation_frame(i):
    # print(i)
    data_m = sex_1.iloc[int(i):int(i+1)] #select data range
    data_f = sex_2.iloc[int(i):int(i+1)] 
    plt.barh(data_m['age'],data_m['value'],color='lightblue',height=1,alpha=0.8)
    if i>16:
        ax.text(0, 15, "Gen Z",ha='center',bbox=box_style,**hfont)
    if i>30:
        ax.text(0, 30, "Millenials",ha='center',bbox=box_style,**hfont)    
    if i>47:
        ax.text(0, 47, "Gen X",ha='center',bbox=box_style,**hfont)
    if i>64:
        ax.text(0, 64, "Baby Boomers",ha='center',bbox=box_style,**hfont)
    if i>83:
        ax.text(0, 83, "Silent Generation",ha='center',bbox=box_style,**hfont)

    return plt.barh(data_f['age'],data_f['value'],color='darkorange',height=1,alpha=0.8)
    
    #chrt2.datavalues(data_f['age'],sex_1['value'])   


animation1 =matplotlib.animation.FuncAnimation(fig, func=animation_frame,frames=91,blit=True)
animation1.save('PopPyramid2.mov', writer=writer)
fig.figure.savefig('popy.svg')
fig.figure.savefig('popy.png')



#####Second we can look at how the average age has changed over time
###generate table with average age and year
### to do this sumproduct is needed


### data manipulation
#av_year = little_frame

av_year_f = little_frame[little_frame['sex']==2]
av_year_m = little_frame[little_frame['sex']==1]

def sumprod_table(input_table): ###function to return a sumproduct of the input table
    input_table = input_table.drop(columns=['sex'])
    input_table['product1']=input_table['age']*input_table['value']
    input_table = input_table.pivot_table(index = ['year'], values=['value','product1'],aggfunc='sum')
    input_table['mean_age']=input_table['product1']/input_table['value']
    input_table = input_table.drop(columns=['product1','value'])
    input_table.reset_index()
    return input_table


av_year = sumprod_table(little_frame)
av_year_f=sumprod_table(av_year_f)
av_year_m=sumprod_table(av_year_m)

#check that all looks right
#av_year.plot(y='mean_age',use_index=True)
#av_year_f.plot(y='mean_age',use_index=True)
#av_year_m.plot(y='mean_age',use_index=True)

av_year = av_year.reset_index()
av_year.columns
av_year['year']=av_year['year'].astype(float) #need to be floats for later use
av_year['year']=round(av_year['year'],8)

def interp_df(df,step): #for nice smooth transistions in the animation data between points need to be interpolated
    newdf1 = pd.DataFrame(np.arange(min(df.iloc[:, 0]),max(df.iloc[:, 0])+step, step).tolist())
    newdf1[0]=newdf1[0].astype(float)
    newdf1[0]=round(newdf1[0],8)
    av_year['year']=round(av_year['year'],8)
    newthing = pd.merge(newdf1,df,how='left', left_on=newdf1.columns[0], right_on=df.columns[0])
    newthing = newthing.interpolate(method='linear')
    newthing=newthing.drop(columns=newthing.columns[1])
    return newthing

interped = interp_df(av_year,0.1)
interped.columns=['year','mean_age']
#av_year['mean_age']=av_year['mean_age'].astype(float)


av_year_f = av_year_f.reset_index()
av_year_f['year']=av_year_f['year'].astype(float)
av_year_f['year']=round(av_year_f['year'],8)

av_year_m = av_year_m.reset_index()
av_year_m['year']=av_year_m['year'].astype(float)
av_year_m['year']=round(av_year_m['year'],8)

interped_f = interp_df(av_year_f,0.1)
interped_f.columns=['year','mean_age']
interped_m = interp_df(av_year_m,0.1)
interped_m.columns=['year','mean_age']


"""~~~~~~~~~~~~~~~~~~~~~~~¦"""
"""      Animations          """
"""~~~~~~~~~~~~~~~~~~~~~~~"""

fig, ax = plt.subplots(dpi=300)
plt.style.use('seaborn-whitegrid')
ax.set_xlim(np.min(interped['year']),np.max(av_year['year']))
ax.set_ylim(np.min(interped_m['mean_age']),np.max(interped_f['mean_age']))
ax.set_title('Average Age of UK Population 2001-2019')
ax.set_xlabel('Year')
ax.set_ylabel('Average Age')
line,=ax.plot(0,0,color='darkgray',linewidth=4)
line2,=ax.plot(0,0,marker='o',ls="",markersize=6,color='black')
line_m,=ax.plot(0,0,color='lightblue',linewidth=6)
line_f,=ax.plot(0,0,color='orange',linewidth=6)





def animation_frame(i):
    #print(i) ### constantly update console to check all running smoothly
    k=0
    if i>180:
        k=-180+i
        i=180  
    if k==1:
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        ax.add_artist( #adding female icon to axes
                AnnotationBbox(
                    OffsetImage(im,zoom=0.2)
                    , (2002, 40.5)
                    , frameon=False
                ) 
        )
        ax.add_artist( #adding male icon to axes
                AnnotationBbox(
                    OffsetImage(im1,zoom=0.2)
                    , (2002, 38)
                    , frameon=False
                ) 
        )

    
    j=math.floor((i/10))
    data = interped.iloc[:int(i+1)] #select data range
    data=data.reset_index()
    line.set_xdata(data['year'])
    line.set_ydata(data['mean_age'])
    data2 = av_year.iloc[:int(j+1)]
    data2=data2.reset_index()
    line2.set_xdata(data2['year'].astype(float))
    line2.set_ydata(data2['mean_age'].astype(float))
    
    
    data3 = interped_f.iloc[:int(k+1)] #select data range
    #data=data.reset_index()
    line_f.set_xdata(data3['year'].astype(float))
    line_f.set_ydata(data3['mean_age'].astype(float))
    
    data4 = interped_m.iloc[:int(k+1)]
    #data2=data2.reset_index()
    line_m.set_xdata(data4['year'].astype(float))
    line_m.set_ydata(data4['mean_age'].astype(float))
    
    return line,line_m,line_f,line2

animation1 =matplotlib.animation.FuncAnimation(fig, func=animation_frame,frames=360,blit=True,repeat=False)
animation1.save('somethingElse.mov', writer=writer)
fig.figure.savefig('Average_age.svg')
fig.figure.savefig('Average_age.png')




### Start looking at LAU locations
### Bring in a shapefile
ecode_bound = gpd.read_file("Data/Districts/Local_Authority_Districts__April_2019__UK_BGC_v2.shp")
pd.options.mode.chained_assignment = None  # default='warn'


#### select only 2019 data
LAUs = big_frame
LAUs = LAUs[LAUs["year"] == '2019'] 
#### drop anything not needed
LAUs = LAUs.drop(columns=['country','sex','year'])
LAUs['product1']=LAUs['age']*LAUs['value']
LAUs=LAUs.drop(columns='age')
####group by Local area
LAUs = LAUs.pivot_table(index = ['geogcode','geogname'], values=['value','product1'],aggfunc='sum')
LAUs['mean_age']=LAUs['product1']/LAUs['value']
LAUs = LAUs.drop(columns=['value','product1'])

###join onto shapefile and plot

ecodebounded = pd.merge(ecode_bound,LAUs,how='left', left_on='LAD19NM', right_on='geogname')

pltmap = ecodebounded.plot(column='mean_age',legend=True,cmap='Greens')
pltmap.set_axis_off();
pltmap.set_title('Average Age in Local Authority Areas')
pltmap.figure.savefig('pltmap.svg')
pltmap.figure.savefig('pltmap.svg')


#pltmap = ecodebounded.plot(column='mean_age',legend=False,cmap='Oranges')
#pltmap.set_axis_off();
#pltmap.figure.savefig('pltmap_noL.svg')
ecodebounded_stats = ecodebounded[['LAD19NM','mean_age']]
ecodebounded_stats.to_csv(r'G:/Documents/Data analysis/Age UK dist/Age Dist Spider/agedist/meanage.csv',index=False)
ecodebounded_stats.to_pickle(path='G:/Documents/Data analysis/Age UK dist/Age Dist Spider/agedist/pickledmeanage',protocol=4)

#Create a list of the top and bottom 10 places
def print_top_bottom(txt1,df,col1):
    print(f"10 areas with the highest {txt1} \n {df.nlargest(10,col1)}")
    print(f"10 areas with the lowest {txt1} \n {df.nsmallest(10,col1)}")
print_top_bottom('average age',ecodebounded_stats,'mean_age') 

lowest_age=ecodebounded_stats.nsmallest(10,'mean_age')

lowest_list=lowest_age.iloc[:,1:2].values.tolist()
lowest_list=lowest_age['LAD19NM'].values.tolist()
lowest_list_number=lowest_age['mean_age'].values.tolist()
age_list=ecodebounded['mean_age'].tolist()
arranged_LAD = ecodebounded['LAD19NM'].tolist()
age_list=age_list
lowest_list=lowest_list[::-1]
age_list=lowest_list_number[::-1]
print(lowest_list)

highest_age=ecodebounded_stats.nlargest(10,'mean_age')
highest_age=highest_age.sort_values('mean_age')
highest_list=highest_age['LAD19NM'].values.tolist()
highest_list_number=  highest_age['mean_age'].values.tolist()


