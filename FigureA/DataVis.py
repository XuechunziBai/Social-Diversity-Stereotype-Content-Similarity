
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(20201010)

##==========================##
## study 1: contextual- level (country and state).
df = pd.read_excel("DataVis.xlsx", sheet_name = 0) # context spreadsheet
df.head(3)

# country plot.
df_country = df.loc[df['dataset'] == 'country']

labels = df_country['group']
x = df_country['herfindahl']
y = df_country['meandist']
area = (df['meandist']*20)**2  # area prop to dispersion
plt.title("Diversity and dispersion in 46 nations and regions")
plt.xlabel("Country-level ethnic diversity")
plt.ylabel("Country-level stereotype dispersion")

plt.scatter(x, y, s=area, alpha=0.5)

for label, x, y in zip(labels, x, y):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-10, 10),
        textcoords='offset points', ha='center', va='bottom',
        bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.2),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'),
		size='small')

plt.show()

## or alternative way.
df_country = df.loc[df['Country'] == 'Lebanon']
plt.figure(figsize=(6,6))
ax=sns.scatterplot(x='Competence',y='Warmth',data=df_country, s=80)
sns.despine(offset=10, trim=False)
ax.set(xlim=(1,5))
ax.set(ylim=(1,5))
ax.set_title('Lebanon',  fontsize=16)
plt.show()



# state plot.
df_country = df.loc[df['dataset'] == 'state']

labels = df_country['group']
x = df_country['herfindahl']
y = df_country['meandist']
area = (df['meandist']*30)**2  # area prop to dispersion
plt.title("Diversity and dispersion in 50 states in the US")
plt.xlabel("State-level objective diversity")
plt.ylabel("Stereotype dispersion")

plt.scatter(x, y, s=area, alpha=0.5)

for label, x, y in zip(labels, x, y):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-10, 10),
        textcoords='offset points', ha='center', va='bottom',
        bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.2),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'),
		size='small')

plt.show()

## or alternative way.
df_country = df.loc[df['State'] == 'Wisconsin']
plt.figure(figsize=(6,6))
ax=sns.scatterplot(x='Competence',y='Warmth',data=df_country, s=80)
sns.despine(offset=10, trim=False)
ax.set(xlim=(1,5))
ax.set(ylim=(1,5))
ax.set_title('Wisconsin', fontsize=16)
plt.show()

##==========================##
## study 2: individual-level.
df = pd.read_excel("DataVis.xlsx", sheet_name = 1) # individual spreadsheet
df.head(3)

# Initialize the figure
f, ax = plt.subplots()
sns.set(style="ticks", palette='deep', font_scale = 1.5)
sns.despine(offset=10, trim=True)

# Plot an estimate of central tendency along with a confidence interval
# 1-10
g = sns.lmplot(x="Individual subjective diversity",y="Stereotype dispersion",col="State", data=df[:319], 
		   x_estimator=np.mean, x_ci=95, col_wrap=5,
		   markers = "v", line_kws={'color': 'teal'}, scatter_kws={'color': 'teal'})
g.set_axis_labels("Individual subjective diversity", "Stereotype dispersion")
plt.show()
# 11-20
g = sns.lmplot(x="Individual subjective diversity",y="Stereotype dispersion",col="State", data=df[320:632], 
		   x_estimator=np.mean, x_ci=95, col_wrap=5,
		   markers = "v", line_kws={'color': 'teal'}, scatter_kws={'color': 'teal'})
g.set_axis_labels("Individual subjective diversity", "Stereotype dispersion")
plt.show()
# 21-30
g = sns.lmplot(x="Individual subjective diversity",y="Stereotype dispersion",col="State", data=df[633:938], 
		   x_estimator=np.mean, x_ci=95, col_wrap=5,
		   markers = "v", line_kws={'color': 'teal'}, scatter_kws={'color': 'teal'})
g.set_axis_labels("Individual subjective diversity", "Stereotype dispersion")
plt.show()
# 31-40
g = sns.lmplot(x="Individual subjective diversity",y="Stereotype dispersion",col="State", data=df[939:1237], 
		   x_estimator=np.mean, x_ci=95, col_wrap=5,
		   markers = "v", line_kws={'color': 'teal'}, scatter_kws={'color': 'teal'})
g.set_axis_labels("Individual subjective diversity", "Stereotype dispersion")
plt.show()
# 41-50
g = sns.lmplot(x="Individual subjective diversity",y="Stereotype dispersion",col="State", data=df[1238:], 
		   x_estimator=np.mean, x_ci=95, col_wrap=5,
		   markers = "v", line_kws={'color': 'teal'}, scatter_kws={'color': 'teal'})
g.set_axis_labels("Individual subjective diversity", "Stereotype dispersion")
plt.show()


# Plot an estimate of central tendency along with a confidence interval
# all data, collapsed.
ax=sns.lmplot(x="Individual-level perceived diversity",y="Individual-level stereotype dispersion", data=df, 
		   x_estimator=np.mean, x_ci=95,
		   markers = "v", line_kws={'color': 'grey'}, scatter_kws={'color': 'black'})
sns.set(style="ticks", palette='deep', font_scale = 1.5)
sns.despine(offset=10, trim=True)
ax.set_xlabel('Individual-level perceived diversity', fontsize=20)
ax.set_ylabel('ndividual-level stereotype dispersion', fontsize=20)
#plt.title("Diversity and stereotype dispersion in 1502 American individuals")
plt.show()



##==========================##
## study 3: longitudinal.
# in r, mean and standard error bar plot.
setwd("C:/Users/xb2/Desktop")
library(ggplot2)
library(data.table)
dt.group.summary = fread('test.csv') # longitudinal dataset extract statistics from MLM.
ggplot(dt.group.summary, aes(x=factor(wave), y=meanD, shape=group, group=group)) + 
    geom_pointrange(aes(ymin=meanD-seD, ymax=meanD+seD), colour="black") +
    geom_point(size=5) + # 21 is filled circle
    geom_jitter(position=position_dodge(0.8)) + 
    coord_cartesian(ylim = c(0.30,0.65)) + 
    xlab("(high school at wave 1)            Time            (college senior at wave 5)") +
    ylab("Individual-level stereotype dispersion change") +
    scale_y_continuous(breaks=seq(0, 1, 0.1)) +         # Set tick every 4
    theme_classic() +
    theme(legend.justification=c(1,0),
          legend.position=c(1,0),
          plot.title = element_text(size=18))




##==========================##
## Geographical heatmap - US
# in ipynb.
import plotly
import plotly.graph_objects as go
import pandas as pd

df = pd.read_excel("DataVis.xlsx", sheet_name = 0) # context spreadsheet
df_states = df.loc[df['dataset'] == 'state'] # state plot.
df_states
df_countries = df.loc[df['dataset'] == 'country'] # state plot.
df_countries

##
fig = go.Figure(data=go.Choropleth(
    locations=df_states['code'], # Spatial coordinates
    z = df_states['meandist'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorscale = 'Reds',
    colorbar_title = "Stereotype dispersion",
))
fig.update_layout(
    title_text = '2018 Immigrant Stereotype Dispersion by State',
    geo_scope='usa', # limite map scope to USA
)
fig.show()
plotly.offline.plot(fig, filename='dispersion-us.html')

##
fig = go.Figure(data=go.Choropleth(
    locations=df_states['code'], # Spatial coordinates
    z = df_states['herfindahl'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorscale = 'Blues',
    colorbar_title = "Diversity index",
))
fig.update_layout(
    title_text = '2010 Immigrant Diversity by State',
    geo_scope='usa', # limite map scope to USA
)
fig.show()
plotly.offline.plot(fig, filename='diversity-us.html')

##
fig = go.Figure(data=go.Choropleth(
    locations=df_countries['code'], # Spatial coordinates
    z = df_countries['meandist'].astype(float), # Data to be color-coded
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorscale = 'Reds',
    colorbar_title = "Stereotype dispersion",
))
fig.update_layout(
    title_text = '2000-2018 Social Group Stereotype Dispersion by Country',
    geo_scope='world',
)
fig.show()
plotly.offline.plot(fig, filename='dispersion-world.html')

##
fig = go.Figure(data=go.Choropleth(
    locations=df_countries['code'], # Spatial coordinates
    z = df_countries['herfindahl'].astype(float), # Data to be color-coded
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorscale = 'Blues',
    colorbar_title = "Diversity index",
))
fig.update_layout(
    title_text = '2003 Alesina Diversity Index by Country',
    geo_scope='world',
)
fig.show()
plotly.offline.plot(fig, filename='diversity-world.html')






####===============simpler figure (correlational)
## study 1: contextual- level (country and state).
df = pd.read_excel("DataVis.xlsx", sheet_name = 0) # context spreadsheet
df.head(3)

# black and white; reg line, world
df_country = df.loc[df['dataset'] == 'country']
ax = sns.regplot(x="herfindahl",y="meandist",data=df_country, scatter_kws={"color": "black"}, line_kws={"color": "darkblue"})
#ax.set_title('Diversity and stereotype dispersion in 46 nations', fontsize=20)
ax.set_xlabel('Country-level ethnic diversity', fontsize=20)
ax.set_ylabel('Country-level stereotype dispersion', fontsize=20)
for line in range(0,df_country.shape[0]):
     ax.text(df_country.herfindahl[line]-0.01, df_country.meandist[line]+0.005, df_country.group[line], size='large', color='gray', weight='semibold')
sns.despine(offset=10, trim=True)
plt.show()

# black and white; reg line, us
df_state = df.loc[df['dataset'] == 'state']
ax = sns.regplot(x="herfindahl",y="meandist",data=df_state, scatter_kws={"color": "black"}, line_kws={"color": "darkblue"})
ax.set_title('Diversity and stereotype dispersion in 50 states in the US', fontsize=20)
ax.set_xlabel('State-level ethnic diversity', fontsize=20)
ax.set_ylabel('State-level stereotype dispersion', fontsize=20)
for line in range(46,96):
     ax.text(df_state.herfindahl[line]-0.01, df_state.meandist[line]+0.005, df_state.group[line], size='large', color='gray', weight='semibold')
sns.despine(offset=10, trim=True)
plt.show()

# us scm, 20 immigrant groups.
df = pd.read_excel("DataVis.xlsx", sheet_name = 3)
df.head(3)
dd_us_immigrant=df.groupby(['Group'], as_index=False).mean()
dd_us_immigrant.head(3)
ax = sns.regplot(x="Competence",y="Warmth", scatter_kws={"color": "black"}, fit_reg=False, data=dd_us_immigrant)
#ax.set_title('2018 US nationwide immigrant SCM', fontsize=20)
ax.set_xlabel('Competence', fontsize=20)
ax.set_ylabel('Warmth', fontsize=20)
#ax.set(xlim=(1,5))
#ax.set(ylim=(1,5))
sns.despine(offset=10, trim=False)
for line in range(0,dd_us_immigrant.shape[0]):
     ax.text(dd_us_immigrant.Competence[line], dd_us_immigrant.Warmth[line], dd_us_immigrant.Group[line], size='large', color='gray', weight='semibold')
plt.show()




