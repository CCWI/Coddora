# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from datetime import datetime, time, date, timedelta
import matplotlib.pyplot as plt
from timeit import default_timer as timer

class OccupancySimulator():
    """
    This class allows to simulate occupants in a hypothetical office room.
    It provides the following simulation functions:
        1. randomOccupancy: a trivial simulation of random occupancy by altering states 
                               of presence and abscence and using a random time until each suceeding transition.
        2. lightswitch-2002: simulation according to the LIGHTSWITCH-2002 method by C.F. Reinhart
                                (see https://doi.org/10.1016/j.solener.2004.04.003). 
        3. lightswitch-2002_variational: simulation according to LIGHTSWITCH-2002 but with varying
                                           times for arrival/departure/breaks.
        4. random_moving_wang_2011: determines times of random moving 
                                     following to the Markov chain approach by Wang et al. 2011
                                     (https://doi.org/10.1007/s12273-011-0044-5)
        5. wob_markov_chain : determines window opening and closing actions by the occupants
                               Note that wob_markov_chain should be executed after the final 
                               determination of occupancy, including random moving. Otherwise, 
                               occupancy during window opening/closing may be removed afterwards. 
    A list of the functions to be successively executed can be set to self.apply, 
    then the run() function can be used for their execution.
    self.maxOccupants limits the number of occupants that use the room.
    """
    
    # Mean sojourn times for random moving / window opening
    random_moving_sojourn_bounds_s0 = ( 5, 20)       # 5-20 min away
    random_moving_sojourn_bounds_s1 = (2*60, 8*60)   # 2-8 h staying
    window_opening_sojourn_bounds_s0 = (4*60, 12*60) # 4-12 h until next expected opening action
    window_opening_sojourn_bounds_s1 = (   5,   20)  # 5-20 min opened
    # Settings for variatonal Lightswitch-2002
    lightswitch_var_arrival_bounds = (6, 13)         # between 6:00 and 13:00
    lightswitch_var_lunch_bounds   = (3, 6)          # 3-6 hours after arrival
    lightswitch_var_departure_bounds = (6, 10)       # 6-10 hours after arrival
    lightswitch_var_lunch_duration_bounds = (30, 90) # lunch takes between 30-90 min
    lightswitch_var_break_duration_bounds = (5, 20)  # break takes between 5-20 min
    
   
    def __init__(self, start_date=datetime(2023, 1, 1), simulation_days=5, maxOccupants=1, seed_value=0):

        self.start_date = start_date
        self.simulation_days = simulation_days
        
        self.minOccupants = 0
        self.maxOccupants = maxOccupants

        self.methods = {"randomOccupancy" : self.randomOccupancy,
                        "lightswitch-2002": self.lightswitch_2002,
                        "lightswitch-2002_variational": self.lightswitch_2002_variational,
                        "random_moving_wang_2011": self.random_moving_wang_2011,
                        # Occupant Behavior
                        "wob_markov_chain": self.wob_markov_chain} # window opening behavior
        
        self.apply = ["lightswitch-2002", 
                      "random_moving_wang_2011",
                      "wob_markov_chain",]
        
        self.seed_value = seed_value
        
        self.df = pd.DataFrame()
        self.occupancy = []
        self.windowState = []
        
    def run(self):
        np.random.seed(self.seed_value)
        start = timer()
        # initialization
        self.end_date = self.start_date + timedelta(days=self.simulation_days)
        self.df['Datetime'] = np.arange(self.start_date, self.end_date, timedelta(minutes=1)).astype(datetime)
        self.df['Date'] = self.df['Datetime'].apply(lambda x: x.date())
        #self.df['Time'] = self.df['Datetime'].apply(lambda x: x.time())
        self.df['Timestamp'] = self.df['Datetime'].values.astype('datetime64[s]').astype('float64')
        self.df['Occupants'] = None
        self.occupancy = [0] * len(self.df)
        self.windowState = [0] * len(self.df)
        # simulation
        print("time passed:", round(timer()-start, 2), "sec")
        for m in self.apply:
            print("> running " + m + "...")
            self.methods[m]()
            print("time passed:", round(timer()-start, 2), "sec")
        self.df['Occupants'] = self.occupants
        self.df['Occupancy'] = self.df['Occupants'].apply(lambda x: 1 if x > 0 else 0)
        self.df['WindowState'] = self.windowState
        return self.df
    
    def plotDay(self, date=None):
        if date == None: # pick at random if not selected
            date = self.df['Date'].unique()[np.random.randint(0, self.simulation_days)]
        elif type(date) == int: # pick by simulation day number
            date = self.df['Date'].unique()[date]
        self.df[self.df['Date'] == date].plot.area(x='Datetime', y='Occupants')
        plt.show()      
        
    def plotDay_Window(self, date=None):
        if date == None: # pick at random if not selected
            date = self.df['Date'].unique()[np.random.randint(0, self.simulation_days)]
        elif type(date) == int: # pick by simulation day number
            date = self.df['Date'].unique()[date]
        self.df[self.df['Date'] == date].plot.area(x='Datetime', y='WindowState')
        plt.show()  
    
    def randomOccupancy(self, initialState=0, timeBounds = {0: (1, 60*16),      #absence  1min-16h
                                                            1: (1, 60*8)}):     #presence 1min-8h
        """
        This function is a trivial simulation of random occupancy by altering states of presence and abscence
        and using a random time until each suceeding transition.
        :param timeBounds: dictonary that contains a tuple (minimum, maximum) 
                             for the duration of abscence (0) and presence (1)
        :param initialState: first state when the simulation is started (0=absence, 1=presence)
        """
        state = abs(initialState-1)
        occ = np.array([0] * self.simulation_days * 1440)
        i = 0
        while i < len(self.df):
            state = abs(state-1) # state transition
            t = np.random.randint(timeBounds[state][0], timeBounds[state][1]) # random time until next transition
            occ[i:i+t-1] = state
            i = i+t
        self.occupants = occ
    
    def lightswitch_2002(self, events=None, duration=None, days=None, maxOccupants=None):
        """
        Performs a simulation of occupants according to the LIGHTSWITCH-2002 method by C.F. Reinhart
        (see https://doi.org/10.1016/j.solener.2004.04.003).        """
        
        if events == None:
            events = {
                "arrival":   time(8,  0),
                "departure": time(18, 0),
                "lunch":     time(12, 0),
                "break1":    time(10, 0), 
                "break2":    time(15, 0)
              }
        for e, t in events.items(): # translate event times to minutes
            events[e] = (t.hour * 60 + t.minute)
        if duration == None:
            duration = { # event durations in minutes
                 "lunch":  60,
                 "break1": 15,
                 "break2": 15
              }
        if days == None:
            days = self.simulation_days
        if maxOccupants == None:
            maxOccupants = self.maxOccupants
        
        self.occupants = np.array([0] * days * 1440)
        for p in range(self.minOccupants, maxOccupants):
            occ = np.array([0] * days * 1440)
            for day in range(0, days):
                events_day = {}
                for e, t in events.items():
                    events_day[e] = events[e] + np.random.randint(-15, 15) + day * 1440
                print(events_day)
                # general presence
                for i in range(events_day["arrival"], events_day["departure"]):
                    occ[i] = 1 
                # breaks
                for e in duration.keys():
                    for i in range(events_day[e], events_day[e] + duration[e]):
                        occ[i]=0
            self.occupants += occ
        return self.occupants
    
    def __add_hours(self, t, hours):
        return (datetime.combine(date.today(), t) + timedelta(hours=hours)).time()
    
    def __pick_time_between(self, t1, t2):
        t1 = datetime.combine(date.today(), t1).timestamp() # convert to timestamps
        t2 = datetime.combine(date.today(), t2).timestamp()
        time_between = np.random.uniform(t1, t2) # pick timestamp between
        time_between = datetime.fromtimestamp(time_between).time() # convert back to time
        return time(time_between.hour, time_between.minute) # retun with minute resolution

    def lightswitch_2002_variational(self):
        """
        Performs a simulation of occupants inspired by the LIGHTSWITCH-2002 method by C.F. Reinhart
        (see https://doi.org/10.1016/j.solener.2004.04.003) but randomly varying the (originally hardcoded) 
        times for arrival, breaks and departure. 
        The random times are picked in between the bounds defined as attributes of this class.
        Short breaks (15min in the original method) are only inserted with a possibility of 50%.
        """
        occ_complete = []
        
        for i in range(0, self.simulation_days):
            
            occ = np.array([0] * 1440)
            for p in range(self.minOccupants, self.maxOccupants):
            
                events = {}
                duration = {}
                # occupants arrive between 6:00 and 13:00 (+-15min)
                t_min, t_max = self.lightswitch_var_arrival_bounds
                events["arrival"] = time(np.random.randint(t_min, t_max+1),  0) 
                # take lunch 3-6 hours (+-15min) after arrival
                t_min, t_max = self.lightswitch_var_lunch_bounds
                events["lunch"]   = self.__add_hours(events["arrival"], np.random.randint(t_min, t_max+1)) 
                t_min, t_max = self.lightswitch_var_lunch_duration_bounds
                duration["lunch"] = np.random.randint(t_min, t_max+1) # lunch break takes 30-90 min
                # leave 6-10 hours (+-15min) after arrival
                t_min, t_max = self.lightswitch_var_departure_bounds
                events["departure"] = self.__add_hours(events["arrival"], np.random.randint(t_min, t_max+1))
                t_min, t_max = self.lightswitch_var_break_duration_bounds
                if np.random.randint(0, 2) == 1: # 50% chance of taking a break
                    events["break1"] = self.__pick_time_between(events["arrival"], events["lunch"]) 
                    duration["break1"] = np.random.randint(t_min, t_max+1) # break takes 10-20 min
                if np.random.randint(0, 2) == 1:  # 50% chance of taking a break
                    events["break2"] = self.__pick_time_between(self.__add_hours(events["lunch"], 
                                                                                 duration["lunch"]/60), 
                                                                                 events["departure"]) 
                    duration["break2"] = np.random.randint(t_min, t_max) # break takes 10-20 min
                
                occ += self.lightswitch_2002(events, duration, days=1, maxOccupants=1)
            occ_complete.extend(occ)    
            
        self.occupants = occ_complete
        return self.occupants
        
           
    def __initialize_sojourn_times(self, s0_bounds, s1_bounds):
        """
        Initializes the mean sojourn times used to define the Markov chain transition probabilities.
        The times are randomly picked between pre-defined bounds for s0 and s1.
        """
        s0_min, s0_max = s0_bounds
        s0 =  np.random.randint(s0_min, s0_max+1)
        s1_min, s1_max = s1_bounds
        s1 =  np.random.randint(s1_min, s1_max+1)
        P = [[1-(1/s0),     1/s0],
             [   1/s1,   1-(1/s1)]]
        print("sojourn[0]: {}min, sojourn[1]: {}min".format(int(s0), int(s1)))
        print("P =", P)
        return P
    
        
    def random_moving_wang_2011(self):
        """
        Determines random moving times of occupants during their general presence following the 
        Markov chain approach by Wang et al. 2011 (https://doi.org/10.1007/s12273-011-0044-5).
        """
        occ = self.occupants
        state = 1
        lastTransition = 0
        for p in range(self.minOccupants, self.maxOccupants):
            for i in range(0, len(occ)):
                if i%1440 == 0: # after each day: set new moving behavior
                    print("day " + str(int(i/1440)))
                    P = self.__initialize_sojourn_times(self.random_moving_sojourn_bounds_s0,
                                                        self.random_moving_sojourn_bounds_s1)
                if occ[i] == 0: 
                    # only consider time at work for random moving
                    continue
                if np.random.rand() > P[state][state]:
                    print("transition {}->{} after {}min".format(state, abs(state-1), \
                                                round((i-lastTransition),1)))
                    lastTransition = i
                    state = abs(state-1) # state transition
                if state == 0:
                    occ[i] -= 1
            
            
    def wob_markov_chain(self):
        """
        Uses a Markov chain to determin window events. This equals the approach used for occupant random moving.
        Transitions from 0 (window closed) to 1 (window open) and vice versa depend on the previous state and 
        on probabilities according to predefined mean sojourn times.
        """
        state = 0
        lastTransition = 0
        for i in range(0, len(self.occupants)):
            if i%1440 == 0: # after each day: set new window opening behavior
                print("day " + str(int(i/1440)))
                P = self.__initialize_sojourn_times(self.window_opening_sojourn_bounds_s0, 
                                                    self.window_opening_sojourn_bounds_s1)
            if self.occupants[i] == 0: 
                # changing window states is only possible during presence
                continue
            if np.random.rand() > P[state][state]:
                print("window transition {}->{} after {}min".format(state, abs(state-1), \
                                            round((i-lastTransition),1)))
                lastTransition = i
                state = abs(state-1) # state transition
            self.windowState[i] = state
            