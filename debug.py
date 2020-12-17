import shoji

db = shoji.connect()
ws = db.builds.humandev.Cerebellum
ws[:].groupby("Clusters").first("Clusters")

