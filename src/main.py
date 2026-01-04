"""
Monte Carlo Simulation for FCT 

Note:
This script is provided for readability only.
It depends on external APIs and private credentials and
is not intended to run out-of-the-box.
"""

import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
import requests

#Simulating Bus Routes:
#Defining populations and distances

n = 32 #Number Of Towns

file_path = "/content/drive/MyDrive/FCT_Towns.xlsx"
Towns = pd.read_excel(file_path, header=None) # Import Excel File with Town Names And Populations



#Fill populations vector
populations_ori = np.zeros(n)
for i in range(n):
    populations_ori[i] = Towns[1][i + 1]

populations_dest = np.zeros(n)
for i in range(n):
    if Towns[1][i + 1] > 1000:
        populations_dest[i] = Towns[1][i + 1]




distances = np.zeros((n,n))
API_KEY = ''
endpoint = "https://maps.googleapis.com/maps/api/distancematrix/json"

#Fill distance matrix, a_i,j gives distacne from town i to j.

for i in range(n):
    for j in range(n):
        if i == j:
            distances[i, j] = 0
        elif distances[j, i] != 0:
            distances[i, j] = distances[j, i]
        else:
            origin = Towns[0][i+1]
            destination = Towns[0][j+1]
            params = {
                "origins": origin,
                "destinations": destination,
                "mode": "driving",
                "departure_time": "now",
                "key": API_KEY,
            }
            r = requests.get(endpoint, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()

            element = data["rows"][0]["elements"][0]
            # Distance in km
            distances[i, j] = element["distance"]["value"] / 1000.0


#Defines a matrix of populations a_i,j = 0 if i=j, population of town j otherwise
matrix_populations_ori = np.zeros([n,n])
for i in range(n) :
  for j in range(n) :
    if i!=j : matrix_populations_ori[i,j] = populations_ori[j]

matrix_populations_dest = np.zeros([n,n])
for i in range(n) :
  for j in range(n) :
    if i!=j : matrix_populations_dest[i,j] = populations_dest[j]


#Picks a random town with proability weighted by population of the town.
def pick_town( pops ) :
  totp = sum( pops )
  town, u, accum = 0, np.random.uniform(0,1), pops[0] / totp
  while u > accum :
    town = town + 1
    accum = accum + pops[town] / totp
  return town


#determines the NN route.
def nn_order(start, stops, dists):
    order = [start]
    visited = [start]

    while len(order) < len(stops):
        last = order[-1]
        nearest, nearest_d = None, float('inf')
        for s in stops:
            if s not in visited and dists[last][s] < nearest_d:
                nearest = s
                nearest_d = dists[last][s]
        order.append(nearest)
        visited.append(nearest)
    return order

#Produces Distance for a route
# Defines a matrix of populations a_i,j = 0 if i=j, population of town j otherwise
matrix_populations_ori = np.zeros([n, n])
for i in range(n):
  for j in range(n):
    if i != j:
      matrix_populations_ori[i, j] = populations_ori[j]

matrix_populations_dest = np.zeros([n, n])
for i in range(n):
  for j in range(n):
    if i != j:
      matrix_populations_dest[i, j] = populations_dest[j]

# Picks a random town with probability weighted by population of the town.
def pick_town(pops):
  totp = sum(pops)
  town, u, accum = 0, np.random.uniform(0, 1), pops[0] / totp
  while u > accum:
    town = town + 1
    accum = accum + pops[town] / totp
  return town

# Determines the NN route.
def nn_order(start, stops, dists):
  order = [start]
  visited = [start]

  while len(order) < len(stops):
    last = order[-1]
    nearest, nearest_d = None, float('inf')
    for s in stops:
      if s not in visited and dists[last][s] < nearest_d:
        nearest = s
        nearest_d = dists[last][s]
    order.append(nearest)
    visited.append(nearest)
  return order

# Produces distance for a route
def bus_journey(N, ori_mat, dest_mat, dists):
  trips = np.zeros([N, 2])
  for i in range(N):
    if i == 0:
      trips[i, 0] = 13  # Start at Enniskillen
      trips[i, 1] = pick_town(dest_mat[int(trips[13, 0]), :])
    elif i == N - 1:
      trips[i, 0] = pick_town(populations_ori)
      trips[i, 1] = 13  # End at Enniskillen
    else:
      trips[i, 0] = pick_town(populations_ori)
      trips[i, 1] = pick_town(dest_mat[int(trips[i, 0]), :])

  # Unique origins
  origins = []
  for i in range(N):
    o = int(trips[i, 0])
    if o not in origins:
      origins.append(o)

  # Nearest-neighbor reorder of origins
  origins_order = nn_order(origins[0], origins, dists)

  dist = 0.0
  loc = origins_order[0]
  onboard = []

  def drop_here(stop):
    to_remove = []
    for k in range(N):
      is_on = k in onboard
      if is_on and int(trips[k, 1]) == stop:
        to_remove.append(k)
    for k in to_remove:
      if k in onboard:
        onboard.remove(k)

  def board_here(stop):
    for k in range(N):
      if int(trips[k, 0]) == stop and k not in onboard:
        onboard.append(k)

  drop_here(loc)
  board_here(loc)

  # Visit origins via NN
  for idx in range(1, len(origins_order)):
    nxt = origins_order[idx]
    if nxt != loc:
      dist += dists[int(loc)][int(nxt)]
      loc = nxt
    drop_here(loc)
    board_here(loc)

  remaining_dests = []
  for k in range(N):
    if k in onboard:
      dstop = int(trips[k, 1])
      if dstop not in remaining_dests:
        remaining_dests.append(dstop)

  # Nearest-neighbor over remaining destinations
  if len(remaining_dests) > 0:
    dest_order = nn_order(loc, [loc] + remaining_dests, dists)[1:]
    for dstop in dest_order:
      if dstop != loc:
        dist += dists[int(loc)][int(dstop)]
        loc = dstop
      drop_here(loc)

  return dist


#Figures


# WTT emissions
DSLWTT  = 2.51
FAMEWTT = 0.168
HVOWTT  = 0.0356

# Energy densities
DSLEN   = 41.0
FAMEEN  = 32.7
HVOEN   = 34.4



#Converts Consumption Rates based on energy densities
def Fuel_conv(X, R1, E1, E2):
    E_blend = (1 - X) * E1 + X * E2
    return (R1 * E1) / E_blend


#Generates the values for a journey
def generate_stochastic(_X_unused=None):
    K = bus_journey(16, matrix_populations_ori, matrix_populations_dest, distances)


    DSLCONS = np.random.uniform(0.103, 0.107)


    DSLPRICE  = np.random.triangular(1.31, 1.34, 1.40)
    FAMEPRICE = np.random.triangular(1.40,1.44,1.50)
    HVOPRICE  = np.random.triangular(1.44, 1.51, 1.61)


    DSLTTW = np.random.triangular(0.271,0.282,0.332)

    scenario = {
        "K": K,
        "DSLCONS": DSLCONS,
        "DSLPRICE": DSLPRICE,
        "FAMEPRICE": FAMEPRICE,
        "HVOPRICE": HVOPRICE,
        "DSLTTW": DSLTTW,
    }

    return scenario


# Cost generators

def cost_DSL(scenario):

    K        = scenario["K"]
    cons_dsl = scenario["DSLCONS"]     # L/km
    p_dsl    = scenario["DSLPRICE"]    # £/L or €/L
    return K * cons_dsl * p_dsl


def cost_FAME(scenario, X):
    """
    Cost for a diesel–FAME blend with FAME fraction X (0–1).
    """
    K        = scenario["K"]
    cons_dsl = scenario["DSLCONS"]


    cons_blend = Fuel_conv(X, cons_dsl, DSLEN, FAMEEN)


    p_dsl  = scenario["DSLPRICE"]
    p_fame = scenario["FAMEPRICE"]
    p_blend = (1 - X) * p_dsl + X * p_fame

    return K * cons_blend * p_blend


def cost_HVO(scenario, X):

    K        = scenario["K"]
    cons_dsl = scenario["DSLCONS"]

    cons_blend = Fuel_conv(X, cons_dsl, DSLEN, HVOEN)

    p_dsl = scenario["DSLPRICE"]
    p_hvo = scenario["HVOPRICE"]
    p_blend = (1 - X) * p_dsl + X * p_hvo

    return K * cons_blend * p_blend


#Emission generators (WTW = TTW + WTT)

def emissions_DSL(scenario):

    K        = scenario["K"]
    cons_dsl = scenario["DSLCONS"]   # L/km
    ttw_dsl  = scenario["DSLTTW"]    # kg/km
    wtt_dsl  = DSLWTT                # kg/L

    # WTW per km = TTW + WTT*consumption
    return K * (ttw_dsl + cons_dsl * wtt_dsl)


def emissions_FAME(scenario, X):
    K        = scenario["K"]
    cons_dsl = scenario["DSLCONS"]
    ttw_dsl  = scenario["DSLTTW"]

    cons_blend = Fuel_conv(X, cons_dsl, DSLEN, FAMEEN)

    ttw_blend = (1 - X) * ttw_dsl

    wtt_blend = (1 - X) * DSLWTT + X * FAMEWTT

    return K * (ttw_blend + cons_blend * wtt_blend)


def emissions_HVO(scenario, X):
    K        = scenario["K"]
    cons_dsl = scenario["DSLCONS"]
    ttw_dsl  = scenario["DSLTTW"]

    cons_blend = Fuel_conv(X, cons_dsl, DSLEN, HVOEN)

    ttw_blend = (1 - X) * ttw_dsl
    wtt_blend = (1 - X) * DSLWTT + X * HVOWTT

    return K * (ttw_blend + cons_blend * wtt_blend)


#Histogram function

def Histo(data, w, xmin, xmax):
    edges = np.linspace(xmin, xmax, w + 1)
    counts = np.zeros(w, dtype=int)
    bin_width = (xmax - xmin) / w

    for val in data:
        if val < xmin or val > xmax:
            continue
        placed = False
        for i in range(w - 1):
            if edges[i] <= val < edges[i + 1]:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[w - 1] += 1

    n_in = counts.sum()
    density = counts / (n_in * bin_width)
    return counts, density, edges


#Monte Carlo parameters
N = 10000
w = 100


# Monte Carlo sampling for diesel cost
y_cost = np.zeros(N)
for i in range(N):
    scen = generate_stochastic(0.0)
    y_cost[i] = cost_DSL(scen)

mean_cost = np.mean(y_cost)
std_cost_diesel = np.std(y_cost)
ci_cost_diesel = np.percentile(y_cost, [2.5, 97.5])

print(mean_cost)
print(std_cost_diesel)
print(ci_cost_diesel)


counts, dens, edges = Histo(y_cost, w, 0, 75)
bin_width = edges[1] - edges[0]

xvals = edges[:-1] + bin_width / 2
density = dens

plt.bar(
    xvals, density,
    width=bin_width,
    edgecolor='#4D4D4D',
    color='#8DD3C7',
    alpha=0.7,
    label='Histogram Of Diesel Cost'
)

plt.axvline(
    x=mean_cost,
    color='blue',
    linestyle='--',
    linewidth=1,
    label='Mean Cost'
)

n = counts.sum()
if n > 0:
    p_hat = counts / n

    z95 = 1.96
    se_density = np.sqrt(p_hat * (1 - p_hat) / n) / bin_width
    err = z95 * se_density

    plt.errorbar(
        xvals, density,
        yerr=err,
        fmt='none',
        ecolor='#4D4D4D',
        capsize=1.5
    )

plt.xlabel("Daily Diesel Cost(£)")
plt.ylabel("Probability Density")
plt.legend()
plt.show()



# Monte Carlo sampling for FAME cost
y_cost = np.zeros(N)
for i in range(N):
    scen = generate_stochastic(0.0)
    y_cost[i] = cost_FAME(scen,1)

mean_cost = np.mean(y_cost)
mean_cost = np.mean(y_cost)
std_cost_diesel = np.std(y_cost)
ci_cost_diesel = np.percentile(y_cost, [2.5, 97.5])

print(mean_cost)
print(std_cost_diesel)
print(ci_cost_diesel)

counts, dens, edges = Histo(y_cost, w, 0, 75)
bin_width = edges[1] - edges[0]

xvals = edges[:-1] + bin_width / 2
density = dens

plt.bar(
    xvals, density,
    width=bin_width,
    edgecolor='#4D4D4D',
    color='#8DD3C7',
    alpha=0.7,
    label='Histogram Of FAME Cost'
)

plt.axvline(
    x=mean_cost,
    color='blue',
    linestyle='--',
    linewidth=1,
    label='Mean Cost'
)

n = counts.sum()
if n > 0:
    p_hat = counts / n

    z95 = 1.96
    se_density = np.sqrt(p_hat * (1 - p_hat) / n) / bin_width
    err = z95 * se_density

    plt.errorbar(
        xvals, density,
        yerr=err,
        fmt='none',
        ecolor='#4D4D4D',
        capsize=1.5
    )

plt.xlabel("Daily FAME Cost(£)")
plt.ylabel("Probability Density")
plt.legend()
plt.show()

# Monte Carlo sampling for HVO cost
y_cost = np.zeros(N)
for i in range(N):
    scen = generate_stochastic(0.0)
    y_cost[i] = cost_HVO(scen,1)



counts, dens, edges = Histo(y_cost, w, 0, 75)
bin_width = edges[1] - edges[0]

xvals = edges[:-1] + bin_width / 2
density = dens

plt.bar(
    xvals, density,
    width=bin_width,
    edgecolor='#4D4D4D',
    color='#8DD3C7',
    alpha=0.7,
    label='Histogram Of HVO Cost'
)

plt.axvline(
    x=mean_cost,
    color='blue',
    linestyle='--',
    linewidth=1,
    label='Mean Cost'
)

n = counts.sum()
if n > 0:
    p_hat = counts / n

    z95 = 1.96
    se_density = np.sqrt(p_hat * (1 - p_hat) / n) / bin_width
    err = z95 * se_density

    plt.errorbar(
        xvals, density,
        yerr=err,
        fmt='none',
        ecolor='#4D4D4D',
        capsize=1.5
    )

plt.xlabel("Daily HVO Cost(£)")
plt.ylabel("Probability Density")
plt.legend()
plt.show()



# Monte Carlo sampling for emissions
y_cost = np.zeros(N)
for i in range(N):
    scen = generate_stochastic(0.0)
    y_cost[i] = emissions_DSL(scen)

mean_cost = np.mean(y_cost)
std_cost_diesel = np.std(y_cost)
ci_cost_diesel = np.percentile(y_cost, [2.5, 97.5])

print(mean_cost)
print(std_cost_diesel)
print(ci_cost_diesel)


counts, dens, edges = Histo(y_cost, w, 0, 200)
bin_width = edges[1] - edges[0]

xvals = edges[:-1] + bin_width / 2
density = dens

plt.bar(
    xvals, density,
    width=bin_width,
    edgecolor='#4D4D4D',
    color='#8DD3C7',
    alpha=0.7,
    label='Histogram Of Diesel Emissions'
)

plt.axvline(
    x=mean_cost,
    color='blue',
    linestyle='--',
    linewidth=1,
    label='Mean Emissions'
)

n = counts.sum()
if n > 0:
    p_hat = counts / n

    z95 = 1.96
    se_density = np.sqrt(p_hat * (1 - p_hat) / n) / bin_width
    err = z95 * se_density

    plt.errorbar(
        xvals, density,
        yerr=err,
        fmt='none',
        ecolor='#4D4D4D',
        capsize=1.5
    )

plt.xlabel("Daily Diesel WTW Emissions (kg CO₂e)")
plt.ylabel("Probability Density")
plt.legend()
plt.show()


# Monte Carlo sampling for FAME emissions
y_cost = np.zeros(N)
for i in range(N):
    scen = generate_stochastic(0.0)
    y_cost[i] = emissions_FAME(scen,1)

mean_cost = np.mean(y_cost)
std_cost_diesel = np.std(y_cost)
ci_cost_diesel = np.percentile(y_cost, [2.5, 97.5])

print(mean_cost)
print(std_cost_diesel)
print(ci_cost_diesel)

counts, dens, edges = Histo(y_cost, w, 0, 10)
bin_width = edges[1] - edges[0]

xvals = edges[:-1] + bin_width / 2
density = dens

plt.bar(
    xvals, density,
    width=bin_width,
    edgecolor='#4D4D4D',
    color='#8DD3C7',
    alpha=0.7,
    label='Histogram Of FAME Emissions'
)

plt.axvline(
    x=mean_cost,
    color='blue',
    linestyle='--',
    linewidth=1,
    label='Mean Emissions'
)

n = counts.sum()
if n > 0:
    p_hat = counts / n

    z95 = 1.96
    se_density = np.sqrt(p_hat * (1 - p_hat) / n) / bin_width
    err = z95 * se_density

    plt.errorbar(
        xvals, density,
        yerr=err,
        fmt='none',
        ecolor='#4D4D4D',
        capsize=1.5
    )

plt.xlabel("Daily FAME WTW Emissions (kg CO₂e)")
plt.ylabel("Probability Density")
plt.legend()
plt.show()


# Monte Carlo sampling for HVO emissions
y_cost = np.zeros(N)
for i in range(N):
    scen = generate_stochastic(0.0)
    y_cost[i] = emissions_HVO(scen,1)

mean_cost = np.mean(y_cost)
std_cost_diesel = np.std(y_cost)
ci_cost_diesel = np.percentile(y_cost, [2.5, 97.5])

print(mean_cost)
print(std_cost_diesel)
print(ci_cost_diesel)


counts, dens, edges = Histo(y_cost, w, 0, 2)
bin_width = edges[1] - edges[0]

xvals = edges[:-1] + bin_width / 2
density = dens

plt.bar(
    xvals, density,
    width=bin_width,
    edgecolor='#4D4D4D',
    color='#8DD3C7',
    alpha=0.7,
    label='Histogram Of HVO Emissions'
)

plt.axvline(
    x=mean_cost,
    color='blue',
    linestyle='--',
    linewidth=1,
    label='Mean Emissions'
)

n = counts.sum()
if n > 0:
    p_hat = counts / n

    z95 = 1.96
    se_density = np.sqrt(p_hat * (1 - p_hat) / n) / bin_width
    err = z95 * se_density

    plt.errorbar(
        xvals, density,
        yerr=err,
        fmt='none',
        ecolor='#4D4D4D',
        capsize=1.5
    )

plt.xlabel("Daily HVO WTW Emissions (kg CO₂e)")
plt.ylabel("Probability Density")
plt.legend()
plt.show()


#FAME mixtures costs

X = [0.05, 0.07, 0.10, 0.20, 0.30, 0.50, 1.0]
N = 10000
yvals = [np.zeros(N) for _ in range(len(X))]

for i in range(N):
    scen = generate_stochastic()
    for j, x in enumerate(X):
        yvals[j][i] = cost_FAME(scen, x)

mean_cost = np.mean(yvals[3])
std_cost_diesel = np.std(yvals[3])
ci_cost_diesel = np.percentile(yvals[3], [2.5, 97.5])

print(mean_cost)
print(std_cost_diesel)
print(ci_cost_diesel)

mean_cost = np.mean(yvals[4])
std_cost_diesel = np.std(yvals[4])
ci_cost_diesel = np.percentile(yvals[4], [2.5, 97.5])

print(mean_cost)
print(std_cost_diesel)
print(ci_cost_diesel)


fig, ax = plt.subplots(figsize=(8, 5))

box = ax.boxplot(
    yvals,
    patch_artist=True,
    labels=[f"{int(x*100)}%" for x in X],
    showmeans=True,
    meanline=True,
    showfliers=False
)


box_color = "#8DD3C7"
outline_color = "#4D4D4D"


for patch in box['boxes']:
    patch.set_facecolor(box_color)
    patch.set_edgecolor(outline_color)
    patch.set_linewidth(1.5)


for element in ['whiskers', 'caps', 'medians', 'means']:
    for line in box[element]:
        line.set_color(outline_color)
        line.set_linewidth(1.5)

plt.xlabel("Percentage of FAME")
plt.ylabel("Daily Fuel Cost(£)")
plt.tight_layout()
plt.show()


#Proportions Box plot FAME Emissions

X = [0.05, 0.07, 0.10, 0.20, 0.30, 0.50, 1.0]
N = 10000


yvals = [np.zeros(N) for _ in range(len(X))]

for i in range(N):

    scen = generate_stochastic()


    for j, x in enumerate(X):
        yvals[j][i] = emissions_FAME(scen, x)

mean_cost = np.mean(yvals[3])
std_cost_diesel = np.std(yvals[3])
ci_cost_diesel = np.percentile(yvals[3], [2.5, 97.5])

print(mean_cost)
print(std_cost_diesel)
print(ci_cost_diesel)

mean_cost = np.mean(yvals[4])
std_cost_diesel = np.std(yvals[4])
ci_cost_diesel = np.percentile(yvals[4], [2.5, 97.5])

print(mean_cost)
print(std_cost_diesel)
print(ci_cost_diesel)

fig, ax = plt.subplots(figsize=(8, 5))

box = ax.boxplot(
    yvals,
    patch_artist=True,
    labels=[f"{int(x*100)}%" for x in X],
    showmeans=True,
    meanline=True,
    showfliers=False
)


box_color = "#8DD3C7"
outline_color = "#4D4D4D"


for patch in box['boxes']:
    patch.set_facecolor(box_color)
    patch.set_edgecolor(outline_color)
    patch.set_linewidth(1.5)


for element in ['whiskers', 'caps', 'medians', 'means']:
    for line in box[element]:
        line.set_color(outline_color)
        line.set_linewidth(1.5)

plt.xlabel("Percentage of FAME")
plt.ylabel("Daily WTW Emissions (kg CO₂e)")
plt.tight_layout()
plt.show()


#Proportions Box plot HVO COST

X = [0.05, 0.07, 0.10, 0.20, 0.30, 0.50, 1.0]
N = 10000


yvals = [np.zeros(N) for _ in range(len(X))]

for i in range(N):
    scen = generate_stochastic()
    for j, x in enumerate(X):
        yvals[j][i] = cost_HVO(scen, x)

mean_cost = np.mean(yvals[3])
std_cost_diesel = np.std(yvals[3])
ci_cost_diesel = np.percentile(yvals[3], [2.5, 97.5])

print(mean_cost)
print(std_cost_diesel)
print(ci_cost_diesel)

mean_cost = np.mean(yvals[4])
std_cost_diesel = np.std(yvals[4])
ci_cost_diesel = np.percentile(yvals[4], [2.5, 97.5])

print(mean_cost)
print(std_cost_diesel)
print(ci_cost_diesel)

fig, ax = plt.subplots(figsize=(8, 5))

box = ax.boxplot(
    yvals,
    patch_artist=True,
    labels=[f"{int(x*100)}%" for x in X],
    showmeans=True,
    meanline=True,
    showfliers=False
)


box_color = "#8DD3C7"
outline_color = "#4D4D4D"


for patch in box['boxes']:
    patch.set_facecolor(box_color)
    patch.set_edgecolor(outline_color)
    patch.set_linewidth(1.5)


for element in ['whiskers', 'caps', 'medians', 'means']:
    for line in box[element]:
        line.set_color(outline_color)
        line.set_linewidth(1.5)

plt.xlabel("Percentage of HVO")
plt.ylabel("Daily Fuel Cost(£)")
plt.tight_layout()
plt.show()

X = [0.05, 0.07, 0.10, 0.20, 0.30, 0.50, 1.0]
N = 10000


yvals = [np.zeros(N) for _ in range(len(X))]

for i in range(N):

    scen = generate_stochastic()


    for j, x in enumerate(X):
        yvals[j][i] = emissions_HVO(scen, x)
mean_cost = np.mean(yvals[3])
std_cost_diesel = np.std(yvals[3])
ci_cost_diesel = np.percentile(yvals[3], [2.5, 97.5])

print(mean_cost)
print(std_cost_diesel)
print(ci_cost_diesel)

mean_cost = np.mean(yvals[4])
std_cost_diesel = np.std(yvals[4])
ci_cost_diesel = np.percentile(yvals[4], [2.5, 97.5])

print(mean_cost)
print(std_cost_diesel)
print(ci_cost_diesel)


fig, ax = plt.subplots(figsize=(8, 5))

box = ax.boxplot(
    yvals,
    patch_artist=True,
    labels=[f"{int(x*100)}%" for x in X],
    showmeans=True,
    meanline=True,
    showfliers=False
)


box_color = "#8DD3C7"
outline_color = "#4D4D4D"


for patch in box['boxes']:
    patch.set_facecolor(box_color)
    patch.set_edgecolor(outline_color)
    patch.set_linewidth(1.5)


for element in ['whiskers', 'caps', 'medians', 'means']:
    for line in box[element]:
        line.set_color(outline_color)
        line.set_linewidth(1.5)

plt.xlabel("Percentage of HVO")
plt.ylabel("Daily WTW Emissions (kg CO₂e)")
plt.tight_layout()
plt.show()


# Mean values
emissions = np.array([98.64, 3.92, 0.79, 81.60, 80.59, 72.56, 71.20])
cost = np.array([25.06, 33.81, 33.65, 26.63, 26.48, 27.40, 27.26])

labels = ["Diesel", "FAME100", "HVO100", "FAME20", "HVO20", "FAME30", "HVO30"]
fuel_type = ["Diesel", "FAME", "HVO", "FAME", "HVO", "FAME", "HVO"]

# Clean professional palette
color_map = {"Diesel": "#000000", "FAME": "#8DD3C7", "HVO": "#0072B2"}
marker_map = {"Diesel": "o", "FAME": "o", "HVO": "o"}

# Light grey for error bars
error_color = "darkgrey"

# 95% CI data
cost_CI = np.array([
    [15.09, 35.74],
    [20.18, 48.31],
    [20.03, 47.82],
    [16.23, 37.85],
    [15.93, 37.61],
    [16.72, 38.91],
    [16.41, 38.70]
])

em_CI = np.array([
    [59.37, 139.52],
    [2.35, 5.60],
    [0.48, 1.12],
    [48.93, 116.31],
    [48.20, 115.08],
    [43.51, 103.42],
    [42.57, 101.60]
])

# Convert CI → error bars
cost_err_lower = cost - cost_CI[:,0]
cost_err_upper = cost_CI[:,1] - cost
em_err_lower = emissions - em_CI[:,0]
em_err_upper = em_CI[:,1] - emissions

plt.figure(figsize=(8,6))

# Plot each point + dashed/light error bars
for i, (e, c, lbl, f) in enumerate(zip(emissions, cost, labels, fuel_type)):
    plt.errorbar(
        e, c,
        xerr=[[em_err_lower[i]], [em_err_upper[i]]],
        yerr=[[cost_err_lower[i]], [cost_err_upper[i]]],
        fmt=marker_map[f],
        markersize=8,
        color=color_map[f],
        ecolor=error_color,
        elinewidth=1,
        capsize=3,
        linestyle='--',
        label=lbl if f == "Diesel" else None
    )


# Diesel point
diesel_x = emissions[0]
diesel_y = cost[0]

# Extract FAME points + diesel
fame_mask = np.array(fuel_type) == "FAME"
fame_x = np.concatenate(([diesel_x], emissions[fame_mask]))
fame_y = np.concatenate(([diesel_y], cost[fame_mask]))

# Extract HVO points + diesel
hvo_mask = np.array(fuel_type) == "HVO"
hvo_x = np.concatenate(([diesel_x], emissions[hvo_mask]))
hvo_y = np.concatenate(([diesel_y], cost[hvo_mask]))

# Fit regression lines (including diesel)
fame_fit = np.polyfit(fame_x, fame_y, 1)
hvo_fit = np.polyfit(hvo_x, hvo_y, 1)

# Generate smooth x-points for lines
x_line_fame = np.linspace(min(fame_x), max(fame_x), 50)
x_line_hvo = np.linspace(min(hvo_x), max(hvo_x), 50)

# Plot best-fit lines
plt.plot(x_line_fame, np.polyval(fame_fit, x_line_fame),
         color="#8DD3C7", linestyle="-", linewidth=2,
         label="FAME trend")

plt.plot(x_line_hvo, np.polyval(hvo_fit, x_line_hvo),
         color="#0072B2", linestyle="-", linewidth=2,
         label="HVO trend")


plt.scatter([], [], color="#8DD3C7", marker="o", label="FAME")
plt.scatter([], [], color="#0072B2", marker="o", label="HVO")
plt.scatter([], [], color="#000000", marker="o", label="Diesel")

plt.xlabel("Mean Daily WTW Emissions (kg CO$_2$e)")
plt.ylabel("Mean Daily Cost (£)")

plt.legend(fontsize=9)
plt.tight_layout()
plt.show()


