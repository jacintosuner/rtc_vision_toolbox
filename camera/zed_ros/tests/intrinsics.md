ZED-X mini
K (rostopic):
[740.16,    0.0, 963.90, 
    0.0, 740.16, 543.87, 
    0.0,    0.0,    1.0]

K (Jiahe's code):
[741.0,    0.00, 963.90,
   0.0,  741.00, 543.88,
   0.0,     0.0,    1.0]
   
ZED-X
K (rostopic):
[734.15,    0.0, 1006.00, 
    0.0, 734.15,  531.73, 
    0.0,    0.0,     1.0]

K (Jiahe's code):
[751.49,    0.0,  942.09,
    0.0, 751.49,  565.95,
    0.0,    0.0,    1.0]


ZED X
focal length from rostopic = 734.15
focal length from Jiahe's code = 751.49

ZED X mini
focal length from rostopic = 740.16
focal length from Jiahe's code = 741.0

**disparity to depth**
```
focal_length = focal_length * 2.0 /3.0
depth = (camera_seperation * focal_length) / disparity
```

https://medium.com/analytics-vidhya/distance-estimation-cf2f2fd709d8
http://www.cs.toronto.edu/~fidler/slides/2015/CSC420/lecture12_hres.pdf