    def daily_weight_fit(self, date):
        """ 
        Update function to the weights of the dynamic system, this will be called 
        once a day after the data has arrived. 
        """

        # # comments in cvx code
        # A = cvx.Variable(self.state_weights.shape)
        # B = cvx.Variable(self.input_weights.shape)

        m = GEKKO(remote = False)
        A = m.Array(
                m.Var, 
                (self.state_weights.shape),
                value = .5,
                lb = -1,
                ub = 1
            )

        ## diagonal entries
        # 0, 0
        A[0][0].value = -.5; A[0][0].lower = -1; A[0][0].upper = 1
        A[1][1].value = -.5; A[1][1].lower = -1; A[1][1].upper = 1
        A[2][2].value = -.5; A[2][2].lower = -1; A[2][2].upper = 1
        A[3][3].value = -.5; A[3][3].lower = -1; A[3][3].upper = 1

        ## non-zero entries
        A[0][3].value = random.random(); A[0][3].upper = 1; A[0][3].lower = -1 # for the beta labelled 4, 1 
        A[1][2].value = random.random(); A[1][2].upper = 1; A[1][2].lower = -1 # beta32
        A[2][0].value = random.random(); A[2][0].upper = 1; A[2][0].lower = -1 # beta13         
        A[1][2].value = random.random(); A[1][2].upper = 1; A[1][2].lower = -1 # beta23
        A[3][2].value = random.random(); A[3][2].upper = 1; A[3][2].lower = -1 # beta34

        ## zero entries 
        A[0][1].value = A[0][1].upper = A[0][1].lower = 0
        A[0][2].value = A[0][2].upper = A[0][2].lower = 0
        A[1][0].value = A[1][0].upper = A[1][0].lower = 0
        A[1][3].value = A[1][3].upper = A[1][3].lower = 0
        A[2][1].value = A[2][1].upper = A[2][1].lower = 0
        A[2][3].value = A[2][3].upper = A[2][3].lower = 0
        A[3][0].value = A[3][0].upper = A[3][0].lower = 0
        A[3][1].value = A[3][1].upper = A[3][1].lower = 0

        ############ B matrix
   
        B = m.Array(
                m.Var, 
                (self.input_weights.shape), 
            )

        # 1 entries: 
        # B_11 and B_21
        B[0][0].value = 1; B[0][0].upper = 1; B[0][0].lower = 1
        B[1][0].value = 1; B[1][0].upper = 1; B[1][0].lower = 1

        # weights w1, w2, w3, w4,..., w7
        B[0][1].value = random.random(); B[0][1].upper = 1; B[0][1].lower = -1
        B[1][2].value = random.random(); B[1][2].upper = 1; B[1][2].lower = -1
        B[2][3].value = random.random(); B[2][3].upper = 1; B[2][3].lower = -1
        B[2][4].value = random.random(); B[2][4].upper = 1; B[2][4].lower = -1
        B[2][5].value = random.random(); B[2][5].upper = 1; B[2][5].lower = -1
        B[2][6].value = random.random(); B[2][6].upper = 1; B[2][6].lower = -1
        B[3][7].value = random.random(); B[3][7].upper = 1; B[3][7].lower = -1

        # zeros
        B[0][2].value = B[0][2].upper = B[0][2].lower = 0
        B[0][3].value = B[0][3].upper = B[0][3].lower = 0
        B[0][4].value = B[0][4].upper = B[0][4].lower = 0
        B[0][5].value = B[0][5].upper = B[0][5].lower = 0
        B[0][6].value = B[0][6].upper = B[0][6].lower = 0
        B[0][7].value = B[0][7].upper = B[0][7].lower = 0
        B[1][1].value = B[1][1].upper = B[1][1].lower = 0
        B[1][3].value = B[1][3].upper = B[1][3].lower = 0
        B[1][4].value = B[1][4].upper = B[1][4].lower = 0
        B[1][5].value = B[1][5].upper = B[1][5].lower = 0
        B[1][6].value = B[1][6].upper = B[1][6].lower = 0
        B[1][7].value = B[1][7].upper = B[1][7].lower = 0
        B[2][0].value = B[2][0].upper = B[2][0].lower = 0
        B[2][1].value = B[2][1].upper = B[2][1].lower = 0
        B[2][2].value = B[2][2].upper = B[2][2].lower = 0
        B[2][7].value = B[2][7].upper = B[2][7].lower = 0
        B[3][0].value = B[3][0].upper = B[3][0].lower = 0
        B[3][1].value = B[3][1].upper = B[3][1].lower = 0
        B[3][2].value = B[3][2].upper = B[3][2].lower = 0
        B[3][3].value = B[3][3].upper = B[3][3].lower = 0
        B[3][4].value = B[3][4].upper = B[3][4].lower = 0
        B[3][5].value = B[3][5].upper = B[3][5].lower = 0
        B[3][6].value = B[3][6].upper = B[3][6].lower = 0

        # dates = pd.date_range(start=self.starting_date, end=date)
        dates = date
        hours = list(range(24))


        # y = [[self.get_days_energy(date = date)] for date in dates]
        y = self.get_days_energy(date = date)
        # baseline_energies = [self.get_hourly_baseline_for_day(date) for date in dates]
        baseline_energies = self.get_hourly_baseline_for_day(date)

        flat_baseline_energies = np.reshape(baseline_energies, -1)
        flat_y = np.reshape(y, -1)

        flat_diff = np.subtract(flat_y, flat_baseline_energies).values

        ## TODO: subtract the baseline energy for that day from y
        timesteps = len(flat_y)

        # TODO: ask Alex, should this be a Param? Or Const? Or nothing?  
        u = [self.get_exogenous_inputs_of_day(date) for hour in hours]
        # u = m.Array(m.Param, 
        #         (len(self.get_exogenous_inputs_of_day(date)), len(hours)), 
        #         [(self.get_exogenous_inputs_of_day(date) for hour in hours)]
        #     )
            

        # # z should be all latent states (check with Alex)
        # z = cvx.Variable((timesteps, 4))

        z = m.Array(m.Var, (timesteps, 4), lb = -1, ub = 1)

        # # c should be (0,0,c,0)
        
        C = np.array([0, 0, 1, 0])
        gammas = m.Array(m.Var, (timesteps-1), lb = -5, ub = 5)

        m.Obj(
            m.sqrt(
                m.sum([(flat_diff[i] - z[i][3])**2 for i in range(len(flat_y))] + 
                    [gammas[i] * 
                        (z[i+1][j] - np.dot(A, z[i])[j] - np.dot(B, u[i])[j])
                            for i in range(timesteps - 1)
                        for j in range(len(z[i])) 
                    ])
                )
            )

        m.options.solver = 2
        m.options.MAX_ITER = 1000
        m.solve()

        # IPython.embed()
        
        return A, B, z
        # return np.array(A.value), np.array(B.value), np.array(z.value)



