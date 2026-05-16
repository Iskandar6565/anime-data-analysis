

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm, binom, uniform, poisson
anime = pd.read_csv("anime_analysis/anime_dataset.csv")
print(anime.columns)

#Task 1  Isolate full-length television series 
#that have established a significant fan base

only_tv = anime[(anime["type"]== "TV") & (anime["members"] > 50000)]
tv_anime = only_tv[["title", "score", "members", "year", "episodes", "studios"]]
tv_anime = tv_anime.dropna(subset = ["score", "year", "studios"])

#Analyzing Studio Performance.
#Identify the most dominant studios and see if
#higher production quantity guarantees higher average ratings

top_studios = tv_anime["studios"].value_counts().head(10).index

studios_filtered = tv_anime[tv_anime["studios"].isin(top_studios)]

studio_summary = studios_filtered.groupby("studios")["score"].agg(["count", "mean"])
print(studio_summary)
plt.figure(figsize= (10,6))
studio_summary["mean"].sort_values().plot(kind = "barh", color = "skyblue")
plt.title("Average score of top 10 studios")
plt.xlabel("Average score")
plt.ylabel("Studio Name")
plt.savefig("studio_average_scores.png", bbox_inches = "tight")



#Task 3: Uncover Historical Quality Trends (Grouping & Line Graph)Goal: 
# Track how the quality of anime television productions has evolved over time.

modern_anime = anime[anime["year"] >1990]
anime_grouped = modern_anime.groupby("year")[["score", "episodes"]].agg(["mean"])
plt.figure(figsize= (10,6))
anime_grouped["score"].plot(kind = "line")
plt.title("Average Score")
plt.xlabel("Years")
plt.ylabel("Mean score")
plt.savefig("average_score.png", bbox_inches = "tight")

plt.figure(figsize = (10,6))
anime_grouped["episodes"].plot(kind = "line")
plt.title("Average Episodes")
plt.savefig("average_episode.png", bbox_inches = "tight")


# Task 4: Popularity vs. Rating Dynamics (Scatterplot & Histogram)Goal:
#  Explore whether massive popularity correlates with a higher user rating score.

print(anime["members"].corr(anime["score"]))
anime["log_members"] = np.log(anime["members"])
plt.figure(figsize= (10, 6))
sns.lmplot(x = "log_members", y = "score", data = anime, ci= None)
plt.savefig("score_members_corr.png", bbox_inches = "tight")
top_favourites = anime.sort_values(by = "favorites", ascending = False)
print(top_favourites[["title", "favorites"]].head(5))

plt.figure(figsize= (10,6))
plt.hist(anime["score"], bins = 20, edgecolor= "black", color = "skyblue")
plt.savefig("scores.png", bbox_inches = "tight")

#Task 1: Normal Distribution & Probability Calculations (norm)
#Goal: Mathematically modelanime scores to evaluate 
#the likelihood of a series becoming a critically acclaimed hit.

anime_mean = tv_anime["score"].mean()
anime_std = tv_anime["score"].std()

prob_above_8 = 1- norm.cdf(8.0, anime_mean, anime_std)
top_5_prob = norm.ppf(0.95, anime_mean, anime_std)
print(prob_above_8)
print(top_5_prob)

# Task 2: Binomial Distribution & Simulations (binom). Goal: Simulate streaming platform
#  business outcomes based on the success rate of your dataset.

p = (tv_anime["score"] >7.5).mean()
print("proportion of 7.5 is " )
print(p)
#license company buys random 15animes. 
#find probability that exactly 6 would be highly successful

prob = binom.pmf(4, 15, p)
print(f"probability of getting exactly 4: {prob: .4f}")
sim_results = binom.rvs(15, p, size = 10000)
plt.figure(figsize=(8,5))
sns.histplot(sim_results, discrete = True, color = "orange", edgecolor = "black", stat = "probability")
plt.savefig("binom_sim_results.png", bbox_inches = "tight")

#Continuous Uniform Distribution & Sampling (uniform)
#Goal: Use synthetic simulations to benchmark random browsing 
# behaviors against actual user review trends.

min_year = tv_anime["year"].min()
max_year = tv_anime["year"].max()
print(f"\nDataset Time Range: From {min_year} to {max_year}")
uniform_sim = uniform.rvs(1970, 2026 - 1970, size = 100)
plt.figure(figsize=(8, 5))
sns.histplot(uniform_sim, bins=15, color="lightgreen", edgecolor="black")
plt.title("Simulation: 1,000 Random Selection Years (Uniform)")
plt.xlabel("Simulated Year")
plt.ylabel("Count of Choices")
plt.savefig("uniform_distrib.png", bbox_inches = "tight")
plt.show()
#Task 4: Poisson Distribution Modeling (poisson)
#Goal: Estimate the production pacing of the modern anime industry.

yearly_production_counts = modern_anime["year"].value_counts()
prod_mean = yearly_production_counts.mean()
print(prod_mean)
prob_50_release = poisson.pmf(50, prod_mean)




