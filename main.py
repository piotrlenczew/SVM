

from ucimlrepo import fetch_ucirepo
wine_quality = fetch_ucirepo(id=186)

X = wine_quality.data.features
Y = wine_quality.data.targets

print(wine_quality.metadata)
print(wine_quality.variables)
