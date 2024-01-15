# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from datetime import datetime, time, date, timedelta
import matplotlib.pyplot as plt
from timeit import default_timer as timer

class OccupancySimulator():
    """
    This class allows to simulate occupants in a hypothetical meeting room.
    It provides the following simulation functions:
        1. random_meetings: determines the timeslots for meetings.
        2. random_meeting_attendants: detemines the occupants attending the meetings 
                                        and their arrival/departure times.
                                      Note that this depends on the previous execution of (1.)
        3. random_moving_wang_2011: determines times of random moving 
                                     following to the Markov chain approach by Wang et al. 2011
                                     (https://doi.org/10.1007/s12273-011-0044-5)
        4. wob_markov_chain : determines window opening and closing actions by the occupants
                               Note that wob_markov_chain should be executed after the final 
                               determination of occupancy, including random moving. Otherwise, 
                               occupancy during window opening/closing may be removed afterwards.
    A list of the functions to be successively executed can be set to self.apply, 
    then the run() function can be used for their execution.
    self.maxOccupants limits the number of occupants that use the room.
    """
    
    # Mean sojourn times for random moving / window opening
    random_moving_sojourn_bounds_s0 = ( 5, 20)       # 5-20 min away
    random_moving_sojourn_bounds_s1 = (12*60, 48*60) #  12-48 h staying
    window_opening_sojourn_bounds_s0 = (  60, 4*60)  # 1-4 h until next expected opening action
    window_opening_sojourn_bounds_s1 = (   5,   20)  # 5-20 min opened
    
    def __init__(self, start_date=datetime(2023, 1, 1), simulation_days=5, maxOccupants=1, seed_value=0):

        self.start_date = start_date
        self.simulation_days = simulation_days
        
        self.minOccupants = 0
        self.maxOccupants = maxOccupants

        self.methods = {"random_meetings" : self.random_meetings,
                        "random_meeting_attendants": self.random_meeting_attendants,
                        "random_moving_wang_2011": self.random_moving_wang_2011,
                        # Occupant Behavior
                        "wob_markov_chain": self.wob_markov_chain} # window opening behavior
        
        self.apply = ["random_meetings",
                      "random_meeting_attendants",
                      "random_moving_wang_2011",
                      "wob_markov_chain"]
        
        self.seed_value = seed_value
        
        self.df = pd.DataFrame()
        self.occupancy = []
        self.windowState = []
        
    def run(self):
        """Executes the defined simulation functions."""
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
    
    def random_meetings(self, initialState=0, timeBounds =  {0: ( 5, 60*4),       #absence  5min-4h
                                                            1: (30, 60*2)}):     #presence 30min-2h
        """
        Generates a random schedule of meetings.
        """
        state = abs(initialState-1)
        occ = np.array([0] * self.simulation_days * 1440)
        i = 0
        while i < len(self.df):
            if (i % 1440 < 480) | (i % 1440 > 1080): # meetings can only begin between 8:00 and 18:00
                occ[i] = 0
                i += 1
                continue
            state = abs(state-1) # state transition
            t = np.random.randint(timeBounds[state][0], timeBounds[state][1]) # random time until next transition
            occ[i:i+t-1] = state
            i = i+t
        self.occupants = occ
        return self.occupants
    
    def random_meeting_attendants(self, random_number_of_attendants=True):
        """
        Simulates random occupants attending meetings in a meeting room. 
        Meeting times need to be defined in advance (e.g. by the random_meeting function) 
        and marked by "1" in the self.occupancy array.
        This function then adds multiple occupants and varys their arrival/departure times.
        :param random_number_of_attendants: 
                            if True:
                               attendant number is randomly set between 1 and self.maxOccupants 
                               for each meeting individually
                            if False:
                                attendant number is always set to self.maxOccupants for all meetings
        """
        occ = self.occupants
        for i in range(0, len(occ)):
            if (occ[i] > 0) & (occ[i-1]==0): # meeting begins
                for j in range(i, len(occ)):
                     if (occ[j] == 0) & (occ[j-1] > 0): # search meeting end
                        occ[i:j] = 0 # delete meeting marker
                        if random_number_of_attendants:  # set number of attendants
                            attendants = np.random.randint(1, self.maxOccupants+1)
                        else:
                            attendants = self.maxOccupants
                        for p in range(0, attendants): # attendants arrive
                            arrival = np.random.randint(-10, 11) # between 10min early and 10min late
                            departure = np.random.randint(-10, 11)
                            if i+arrival < j+departure:
                                occ[i+arrival : j+departure] += 1 # stay until the end of the meeting (+-10min)
                            else:
                                 occ[i+arrival : j]
                        break
                i = j
        self.occupants = occ
        return self.occupants
   
    
    def __add_hours(self, t, hours):
        return (datetime.combine(date.today(), t) + timedelta(hours=hours)).time()
    
    def __pick_time_between(self, t1, t2):
        t1 = datetime.combine(date.today(), t1).timestamp() # convert to timestamps
        t2 = datetime.combine(date.today(), t2).timestamp()
        time_between = np.random.uniform(t1, t2) # pick timestamp between
        time_between = datetime.fromtimestamp(time_between).time() # convert back to time
        return time(time_between.hour, time_between.minute) # retun with minute resolution
    
    
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
        