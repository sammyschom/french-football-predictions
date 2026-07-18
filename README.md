A program demonstrating Poisson-based football match and tournament prediction 

A statistical model estimating team scoring rates from historical match data, using them to predict match outcomes and simulate an entire knockout tournament via Monte Carlo methods.

Originally built for the *Analyse de Données Avancées* module during a year studied in France as part of a Physics MSc at the University of Manchester.


How it works:
1. **Estimates each team's scoring rate (λ)** from historical match data, modelling goals 
   scored per match as a Poisson distribution.
2. **Validates the model** by plotting actual goal distributions against the theoretical 
   Poisson distribution for a sample of teams.
3. **Predicts match outcome probabilities** (win/draw/loss) for every possible pairing of 
   teams, by combining both teams' Poisson distributions across all realistic scorelines.
4. **Simulates a full knockout tournament** (Round of 16 → Quarter-Final → Semi-Final → Final) 
   1,000 times using Monte Carlo simulation, tracking how often each team reaches each stage 
   and wins the tournament outright.

The adjustable parameter 'num_simulations' adjusts how many full tournaments are simulated.

Running the program:

To run the program, it is assumed that the csv 'DataFoot.csv' is in the same directory as FOOTPREDICTIONS.py. The program requires Python 3 with numpy, matplotlib, pandas, and scipy installed.
