; Institue of Software Technology
; TU Wien
;
; Self-Organizing Systems
; Assignment 3
; Author: Abdel Aziz Taha
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;



; The additional (customized) properties of the
; patches (grid cells) are to be defined here

patches-own
[
  val  ; val represents the fitness value associated with the patch
       ; the goal of the particle swarm is to find the patch with the best fitness value
]

; The additional (customized) properties of the
; turtles (agents) are to be defined here

turtles-own
[
  vx                  ; velocity vector coordinate x
  vy                  ; velocity in the y direction

  personal-best-val   ; value of personal best
  personal-best-x     ; x coord. of personal best
  personal-best-y     ; x coord. of personal best
]

; global variables are to be defined here

globals
[
  global-best-x    ; x coordinate of global best
  global-best-y    ; y coordinate of global best
  global-best-val  ; value of global best
  global-best-iter

  true-best-patch  ; the patch representing the optimal (real best)

  max-x ;          ; maximum x coordinate
  max-y;           ; maximum y coordinate

  iterations       ; counter for iterations
]

; The function setup initilizes the search landscape and the agents
; for the search. It is called by the button controll setup

to setup
  clear-all
  set iterations 1;
  initialize-topology

  initialize-agents

  initialize-global-best

  ;save this setup in a file for posibly later use. Existing file is overwritten
  ;export-world "backup.txt"

  update-highlight
  reset-ticks

end


; The function initialize-topology initializes a search land scape (topology)
; which represents the fittness function
; each point in the search space (each patch) should be set a value

to initialize-topology
  set max-x max [pxcor] of patches
  set max-y max [pycor] of patches
  ask patches [
     if fitness_function = "Example"
       [set val example_function pxcor pycor]

     if fitness_function = "Random"
       [set val fittness_function_random pxcor pycor]

     if fitness_function  = "Langermann"
       [set val fittness_function_langermann pxcor pycor]

     if fitness_function = "Schwefel"
       [set val fittness_function_schwefel pxcor pycor]

     if fitness_function = "Shubert"
       [set val fittness_function_shubert pxcor pycor]

     if fitness_function  = "Schaffer"
       [set val fittness_function_schaffer pxcor pycor]

     if fitness_function = "Eggholder"
       [set val fittness_function_eggholder pxcor pycor]

     if fitness_function = "Easom"
       [set val fittness_function_easom pxcor pycor]

     if fitness_function = "Booth"
       [set val fittness_function_booth pxcor pycor]

  ]

  let min-val min [val] of patches
  let max-val max [val] of patches

  ask patches [
    ;normalize the values to be between 0 and 1
    set val (val - min-val) / (max-val - min-val)

    ;check whether the patch violates a constrain
    ;if yes check the constraint_handling_method
    ;if rejection method, then set the value to zero and color to red
    ;if penalty method, then make subtract one from the value and make the color orange
    ;otherwise, set the patch color according to its value
    ifelse ((violates pxcor pycor) and constraints = TRUE)
     [
       if (constraint_handling_method = "Rejection Method")
        [
          set val 0
          set pcolor 15
        ]
        if (constraint_handling_method = "Penalty Method")
        [
          set val (val - 1)
          set pcolor 25
        ]
     ]

     [
       set pcolor scale-color gray val 0.0  1
     ]
   ]


     ask max-one-of patches [val]
  [
    set true-best-patch self
  ]

end

; The function creates agents (turtels) depending on the control slider
; swarm size. It intializes these agents with random positions and
; velocities.

to initialize-agents
  ; create particles and place them randomly in the world
  create-turtles population-size
  [
    setxy random-xcor random-ycor

     ;avoid turtles in the regions violating the constraints
      while [(violates xcor ycor) and (constraints = TRUE)]
     [
        set ycor  random-ycor
        set xcor random-xcor
     ]

    ;give the particles normally distributed random initial velocities for both x and y directions
    set vx random-normal 0 1
    set vy random-normal 0 1

    ;the starting spot is the particle's current best location.
    set personal-best-val val
    set personal-best-x xcor
    set personal-best-y ycor

    ;choose a random basic NetLogo color, but not gray
    set color one-of (remove-item 0 base-colors)

    ;make the particles a little more visible
    set size 7

  ]

end

; The global best point coordinates are initialized as the coordinates
; of the patch with best fittness value
;
to initialize-global-best
  set global-best-x  (max [pxcor] of patches)
  set global-best-y   (max [pycor] of patches)
end


; This function iterate is called by the controll buttons "go" and "step"
; it represents the iterations of the PSO algorithm

to iterate

  update-particle-positions

  update-personal-best

  update-global-best

  handle-visualation-options

  update-highlight

  if (global-best-val = [val] of true-best-patch) or (iterations > 49) [
    ; Export hyperparameters and final fitness value to a file before stopping
    export-run-results
    stop
  ]
  
  set iterations (iterations + 1)

  tick

end



to update-particle-positions

  ask turtles
  [
    let vx_bak vx ; backup th particle velocity for constraint handling
    let vy_bak vy ; backup th particle velocity for constraint handling

    let x_bak xcor; backup th particle position for constraint handling
    let y_bak ycor; backup th particle position for constraint handling


    set vx particle-inertia * vx
    set vy particle-inertia * vy

    facexy personal-best-x personal-best-y
    let dist distancexy personal-best-x personal-best-y
    set vx vx +  personal-confidence * (random-float 1.0) * dist * dx
    set vy vy +  personal-confidence * (random-float 1.0) * dist * dy

    ; change my velocity by being attracted to the "global best" value anyone has found so far
    facexy global-best-x global-best-y
    set dist distancexy global-best-x global-best-y
    set vx vx +  swarm-confidence * (random-float 1.0) * dist * dx
    set vy vy +  swarm-confidence * (random-float 1.0) * dist * dy

    ; speed limits are particularly necessary because we are dealing with a toroidal (wrapping) world,
    ; which means that particles can start warping around the world at ridiculous speeds
    if (vx > particle-speed-limit) [ set vx particle-speed-limit ]
    if (vx < 0 - particle-speed-limit) [ set vx 0 - particle-speed-limit ]
    if (vy > particle-speed-limit) [ set vy particle-speed-limit ]
    if (vy < 0 - particle-speed-limit) [ set vy 0 - particle-speed-limit ]

    let x (xcor + vx)
    let y  (ycor + vy)



    ; The Rejection constraint handling is realized here:
    ; If a point violates a constraint, it is rejected
    ; and the velocity and position are reset to the backup values
    ifelse ( (violates x y) and (constraints = TRUE) and (constraint_handling_method = "Rejection Method") )
    [
       set vx -1 * vx_bak
       set vy -1 * vy_bak

       set xcor x_bak
       set ycor y_bak

    ]

    [
       ; face in the direction of my velocity
       facexy (xcor + vx) (ycor + vy)
       ; and move forward by the magnitude of my velocity
       forward sqrt (vx * vx + vy * vy)
    ]

  ]

end


; Updates the "personal best" location for each particle,
to update-personal-best
    ask turtles [
    if val > personal-best-val
    [
      set personal-best-val val
      set personal-best-x xcor
      set personal-best-y ycor
    ]
  ]

end


; Updates the "globa best" location and value.

to update-global-best
  ask max-one-of turtles [personal-best-val]
  [
    if global-best-val < personal-best-val
    [
      set global-best-val personal-best-val
      set global-best-x personal-best-x
      set global-best-y personal-best-y
      set global-best-iter iterations
    ]
  ]
end

; Example function
to-report example_function [x y] ;
  let x1 90 /  max-x * x ; scale x to have a value from -90 to 90
  let y1 180 /  max-y * y ; scale y to have a value from -180 to 180
  report (-1 * (x1 ^ 2 + y1 ^ 2 + 25 * ( sin(x1 + 1) ^ 2 + sin(y1) ^ 2) ) + 20)
end

; dummy random fitness function
to-report fittness_function_random [x y]
  report random-normal 0 1;
end

; Langermann function
; values for m, c & A from https://www.sfu.ca/~ssurjano/langer.html
to-report fittness_function_langermann [x y]
  let x1 (5 /  max-x * x) + 5 ; scale x to have a value from 0 to 10
  let y1 (5 /  max-y * y) + 5 ; scale x to have a value from 0 to 10
  let f1 1 * exp(-(1 / pi) * ((x1 - 3) ^ 2 + (y1 - 5) ^ 2)) * cos(pi * ((x1 - 3) ^ 2 + (y1 - 5) ^ 2))
  let f2 2 * exp(-(1 / pi) * ((x1 - 5) ^ 2 + (y1 - 2) ^ 2)) * cos(pi * ((x1 - 5) ^ 2 + (y1 - 2) ^ 2))
  let f3 5 * exp(-(1 / pi) * ((x1 - 2) ^ 2 + (y1 - 1) ^ 2)) * cos(pi * ((x1 - 2) ^ 2 + (y1 - 1) ^ 2))
  let f4 2 * exp(-(1 / pi) * ((x1 - 1) ^ 2 + (y1 - 4) ^ 2)) * cos(pi * ((x1 - 1) ^ 2 + (y1 - 4) ^ 2))
  let f5 3 * exp(-(1 / pi) * ((x1 - 7) ^ 2 + (y1 - 9) ^ 2)) * cos(pi * ((x1 - 7) ^ 2 + (y1 - 9) ^ 2))
  report (f1 + f2 + f3 + f4 + f5);
end

; Schwefel function
; alpha = 418.9829
; n = 2
to-report fittness_function_schwefel [x y]
  let x1 512 /  max-x * x ; scale x to have a value from -512 to 512
  let y1 512 /  max-y * y ; scale y to have a value from -512 to 512
  report (418.9829 * 2 - x * sin(sqrt(abs(x))) - y * sin(sqrt(abs(y))));
;
end

; Shubert function
to-report fittness_function_shubert [x y]
  let x1 100 /  max-x * x ; scale x to have a value from -100 to 100
  let y1 100 /  max-y * y ; scale y to have a value from -100 to 100
  let f1x 1 * cos((1 + 1) * x1 + 1)
  let f2x 2 * cos((2 + 1) * x1 + 2)
  let f3x 3 * cos((3 + 1) * x1 + 3)
  let f4x 4 * cos((4 + 1) * x1 + 4)
  let f5x 5 * cos((5 + 1) * x1 + 5)
  let f1y 1 * cos((1 + 1) * y1 + 1)
  let f2y 2 * cos((2 + 1) * y1 + 2)
  let f3y 3 * cos((3 + 1) * y1 + 3)
  let f4y 4 * cos((4 + 1) * y1 + 4)
  let f5y 5 * cos((5 + 1) * y1 + 5)
  report ((f1x + f2x + f3x + f4x + f5x) * (f1y + f2y + f3y + f4y + f5y));
end

; Schaffer function
to-report fittness_function_schaffer [x y]
  let x1 100 /  max-x * x ; scale x to have a value from -100 to 100
  let y1 100 /  max-y * y ; scale y to have a value from -100 to 100
  report (0.5 + (sin(x1 ^ 2 - y1 ^ 2) ^ 2 - 0.5) / (1 + 0.001 * (x1 ^ 2 + y1 ^ 2)) ^ 2);
end

; Eggholder function
to-report fittness_function_eggholder [x y]
  let x1 51200 /  max-x * x ; scale x to have a value from -51200 to 51200
  let y1 51200 /  max-y * y ; scale y to have a value from -51200 to 51200
  report (-(y + 47) * sin(sqrt(abs((x1 / 2) + y1 + 47))) - x * sin(sqrt(abs(x1 - y1 - 47))));
end

; Easom function
to-report fittness_function_easom [x y]
  let x1 100 /  max-x * x ; scale x to have a value from -100 to 100
  let y1 100 /  max-y * y ; scale y to have a value from -100 to 100
  report (-1 * (cos(x1) * cos(y1) * exp(-((x1 - pi) ^ 2 + (y1 - pi) ^ 2))));
end

; Booth function
to-report fittness_function_booth [x y]
  let x1 10 /  max-x * x ; scale x to have a value from -10 to 10
  let y1 10 /  max-y * y ; scale y to have a value from -10 to 10
  report (-1 * ((x1 + 2 * y1) ^ 2 + (2 * x1 + y1 - 5) ^ 2));
end


; constraint example
to-report constrain_example [x y]
  report (x ^ 2 > y ^ 2)
end

; dummy random constrinat to be implemented by students
to-report constrain_1 [x y]
  report (x ^ 2 + y ^ 2 < 6000)
end

; dummy random constrinat to be implemented by students
to-report constrain_2 [x y]
  report ((x > 3 * y) or (3 * x < y))
end

; dummy random constrinat to be implemented by students
to-report constrain_3 [x y]
  report ((x > y + 20) or (x < y - 20))
end

; dummy random constrinat to be implemented by students
to-report constrain_4 [x y]
  report (x ^ 2 + y ^ 2 < 9000 and x ^ 2 + y ^ 2 > 4000)
end

; dummy random constrinat to be implemented by students
to-report constrain_5 [x y]
  report x > y
end

; dummy random constrinat to be implemented by students
to-report constrain_6 [x y]
  report 10 * x < y ^ 2
end

; dummy random constrinat to be implemented by students
to-report constrain_7 [x y]
  report tan(2 * x) < tan(4 * y)
end


to-report constrain_8 [x y]
  ifelse sin(8 * x) < sin(8 * y)
  [report TRUE]
  [report FALSE]

end

to-report constrain_9 [x y]
  ifelse sin(x) * sin(y) < 0.2
  [report TRUE]
  [report FALSE]

end

to-report constrain_10 [x y]
  ifelse   tan(x * y) < 1
  [report TRUE]
  [report FALSE]

end


to-report violates [x y]
  if ( Constraint = "Constraint 1")
  [report constrain_1 x y]

  if ( Constraint = "Constraint 2")
  [report constrain_2 x y]

  if ( Constraint = "Constraint 3")
  [report constrain_3 x y]

  if ( Constraint = "Constraint 4")
  [report constrain_4 x y]

  if ( Constraint = "Constraint 5")
  [report constrain_5 x y]

  if ( Constraint = "Constraint 6")
  [report constrain_6 x y]

  if ( Constraint = "Constraint 7")
  [report constrain_7 x y]

  if ( Constraint = "Constraint 8")
  [report constrain_8 x y]

  if ( Constraint = "Constraint 9")
  [report constrain_9 x y]

  if ( Constraint = "Constraint 10")
  [report constrain_10 x y]

  if ( Constraint = "Example")
  [report constrain_example x y]

end

to update-highlight
  ifelse highlight-mode = "Best found"
  [ watch patch global-best-x global-best-y ]
  [
    ifelse highlight-mode = "True best"
    [  watch true-best-patch ]
    [  reset-perspective ]
  ]
end


to load
  clear-all
  import-world (path-to-load)
  update-highlight
  reset-ticks
end

to handle-visualation-options
  ask turtles [
    ifelse trails-mode = "None" [ pen-up ] [ pen-down ]
  ]

  if (trails-mode != "Traces")
    [ clear-drawing ]

end

to show-optimum
 watch true-best-patch
 tick
  wait 0.5
 watch patch global-best-x global-best-y

end

; Function to export hyperparameters and final fitness value
to export-run-results
  ; Constructing hyperparameter details
  let final-results (word "Particle Inertia: " particle-inertia ", "
                       "Personal Confidence: " personal-confidence ", "
                       "Swarm Confidence: " swarm-confidence ", "
                       "Population Size: " population-size ", "
                       "Particle Speed Limit: " particle-speed-limit ", "
                       "Constraint Handling Method: " constraint_handling_method ", "
  										 "Final Fitness: " global-best-val ", "
    									 "Optimum Found After: " global-best-iter ", "
    								   "Iterations: " iterations)

  ; Display the results in the console
  show final-results
end
