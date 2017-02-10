# Fishing Activity Detector
In this project I will collaborate with [Global Fishing Watch](http://www.globalfishingwatch.org/) to detect fishing activity in the ocean using data from satellite Automatic Identification System [AIS](https://en.wikipedia.org/wiki/Automatic_identification_system) collected from different vessels around the world. The AIS data contains the latitude, longitude, speed and course of the vessels at different times.  


## Motivation

Overfishing and illegal fishing are becoming big problems around the world. There are records of intensive and often illegal fishing of West Africa’s waters by Asian and European fleets that reduce the regular catch for the local populations, increasing their poverty levels. In Sierra Leone alone there are two to three times more ships fishing in the country’s near-shore waters than have licenses to do so. 

“Being able to see which vessels are fishing where would be a tremendous help in reducing illegal fishing,” says Josephus Mamie, head of Sierra Leone’s Fisheries Research Unit. [1]

Global Fishing Watch (GFW) is an organization that analyzes data from the Automatic Identification System (AIS), which is collected by satellites and terrestrial receivers, to identify apparent fishing behavior based on the movement of vessels over time.


## Data

AIS system was put in place to guarantee the safety of vessels, it provides collisions avoidance and allow maritime authorities to track and monitor vessel movements. Each vessel periodically reports information including the vessel’s identity, type, position, course, speed, navigational status and other safety-related information.

Vessels fitted with AIS transceivers can be tracked by AIS base stations located along coast lines or, when out of range of terrestrial networks, through a growing number of satellites that are fitted with special AIS receivers. [2] 20M of data points are produced per day this way. But in my case I will have an aggregate of data that has been already labeled by different experts and by graduate students on Dalhousie University in Halifax. 

Training data was kindly made available to me by Mr. David Kroodsma, from SkyTruth one of the collaborating partners.

Notes on the data:

The data features include the type of boat (troller, drifting long lines, fixed gear) and time series of position, speed, and course of the vessels.
The intervals of data acquisition is not constant. There may be gaps when the vessels are not on the satellite’s field of “view”. It takes 90 mins for a low orbit satellite to go around the world, so we can expect gaps ~ 1h. 
When there is good coverage the time intervals still vary from 2 seconds to 2 minutes, due to differences in satellite coverage around the world and signal interference.




## The problem: “Deciding when fishing is happening using AIS data”

The team at Global Fishing Watch needs help with the algorithm that identifies when vessels are fishing. They also would like to have a sense of how well does the algorithm perform in different parts of the world (i.e. less satellite coverage, increased interference near busy ports, etc)

One benchmark result that I expect to pick up from the AIS data is the following heuristic: If the trajectories of the vessel present “wiggles”, or situations in which they stayed for a long time around the same place in the open ocean, it may be because they are fishing. If, on the other hand, the trajectories are straight, the vessels are either cargo ships, passenger vessels, or they are simply no fishing.


## References

[1]  http://www.ipsnews.net/2016/09/new-public-website-offers-detailed-view-of-industrial-fishing/
[2]  http://skytruth.org/, https://en.wikipedia.org/wiki/Automatic_identification_system


