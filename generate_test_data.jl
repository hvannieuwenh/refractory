using Random
using CSV, Tables
using Statistics
using Plots
using JLD

A = ones(5, 5)

#define randomness
random_1 = MersenneTwister(1434)

#generate test data for validation (inverse sampling)
function Generate_Test_Data_J1(num_neurons, num_points, J, lambda_RQ, h_QA, lambda_AR)

    #convert J to population dependant value
    J = J/num_neurons


    #stores state of each nueron
    #state vector 1 (0,1,2), 0 = R, 1 = Q, 2 = A
    #state vector 2 (0,1) 0 = R or Q, A = 1
    state_vector = zeros(Int8,num_neurons,1) 
    state_vector_2 = zeros(Int8,num_neurons,1)


    #keeps track of transition time for each nueron given a constant set of parameters
    #time_vector_local - transition time calculated at each step, lowest gets recorded, the rest are forgotten
    #time_vector_global, keeps track of total time since the last transition of nueron i
    time_vector_local = 0
    time_vector_global = zeros(num_neurons,1)

    #counter for number of active nuerons
    num_active = 0

    #for plotting pi_a
    #active tracker will keep track of number of active at each transitio from QA or AR
    #time tracker, keeps track of the time interval between each
    active_tracker = Vector{Float64}([])
    time_tracker = Vector{Float64}([])

    #keeps track of the time between each state transition for a particular nueron [i]
    time_q = Vector{Vector{Float64}}([[] for k = 1:num_neurons])
    time_a = Vector{Vector{Float64}}([[] for k = 1:num_neurons])
    time_r = Vector{Vector{Float64}}([[] for k = 1:num_neurons])

    time_q_int = [[0,0]] #keeps track of time interval betwee refractory and active 

    #keeps track of the state of all other nuerons at these transitions
    State = Vector{Matrix{Int64}}([state_vector_2 for k = 1:num_neurons])
    State2 = zeros(num_neurons, 1)


    plot_time = 0
    plot_time_running = 0


    #num points is the number of state transitions (specified by the user)
    for i in 1:num_points
        print(" ", i, " ")

        #low_time keeps track of lowest transition time at each interval and the nueron(index)
        #for which this occurs
        low_time = 100000000
        low_time_index = 0



        for j in 1:num_neurons

            #inverse sampling, generates time until transition from exponential distributions
            #parameters depend on the current state

            if (state_vector[j] == 0)
                time_vector_local = -(1/lambda_RQ)*log(1-rand(random_1));

            elseif (state_vector[j] == 1)

                lambda_QA = ((tanh((h_QA + J*(num_active))/2))+1)/2;
                #print(lambda_QA)
                time_vector_local = -(1/lambda_QA)*log(1-rand(random_1));


            else (state_vector[j] == 2)

                time_vector_local = -(1/lambda_AR)*log(1-rand(random_1));

            end


            #record the next nueron that will transition
            if (time_vector_local < low_time)

                low_time = time_vector_local
                low_time_index = j

            end
        end


        #update total time in each state for each nueron
        time_vector_global  = time_vector_global .+ low_time
        plot_time = plot_time .+ low_time
        plot_time_running = plot_time_running .+low_time

        #record total time of transitioning nueron on previous state
        low_time = time_vector_global[low_time_index]
        time_vector_global[low_time_index] = 0



        #push data to correct vector depending on which state it transitioned from
        #update num_active counter
        if state_vector[low_time_index] == 2
            state_vector[low_time_index] = 0

            push!(time_a[low_time_index], low_time)

            state_vector_2[low_time_index] = 0
            num_active -= 1

            #test stuff
            push!(time_tracker, plot_time)
            push!(active_tracker, num_active)
            plot_time = 0

            #new
            State2 = hcat(State2, state_vector_2 )

        elseif state_vector[low_time_index] == 1
            state_vector[low_time_index] += 1

            push!(time_q[low_time_index], low_time)


            state_vector_2[low_time_index] = 1
            State[low_time_index] = hcat(State[low_time_index], state_vector_2)

            #new
            temp_state_vector_2 = state_vector_2
            #temp_state_vector_2[low_time_index] = 0
            State2 = hcat(State2, temp_state_vector_2)

            time_q_start = plot_time_running - low_time - time_r[low_time_index][end] + 1
            time_q_int = vcat(time_q_int, [[plot_time_running - low_time - time_r[low_time_index][end], plot_time_running]])
            
            num_active += 1


            #test stuff
            push!(time_tracker, plot_time)
            push!(active_tracker, num_active)
            plot_time=0
        else
            state_vector[low_time_index] += 1
            push!(time_r[low_time_index], low_time)
        end

    end


    #data for active population plot
    x, y = Plot_Data(time_tracker, active_tracker, num_neurons)


    #process data to csv files
    Process_Data(time_r, time_q, time_a, State, num_neurons)

    return x, y, State2, time_q_int

end

function Process_Data(tr, tq, ta, State, num_neurons)

    for l in 1:num_neurons

        if size(tr[l]) != size(tq[l])
            tr[l] = tr[l][1:end-1]
        end

        t_qr = tq[l] .+ tr[l]

        State[l] = State[l][1:end,2:end]


        CSV.write("Data/Ta_"*string(l, base = 10, pad = 0)*".csv",  Tables.table(ta[l]), writeheader=false)
        CSV.write("Data/Tqr_"*string(l, base = 10, pad = 0)*".csv",  Tables.table(t_qr), writeheader=false)
        CSV.write("Data/S_"*string(l, base = 10, pad = 0)*".csv",  Tables.table(State[l]), writeheader=false)

    end

    return
end

function Plot_Data(time, active, num_nuerons)
    b = reverse(time)
    a = size(b)[1]
    c = [sum(b[i:a]) for i in 1:a]
    c = reverse(c)

    active = active ./ num_nuerons

    return c, active
end

#generate data, (num_neurons, num_points, J, lambda_RQ, h_QA, lambda_AR)
x, y, z, interval = Generate_Test_Data_J1(1000, 8000, 0, 0.01, -5.0, 0.8)

plot(x,y)
#generate state that contains all needed data

state2 = zeros(1000, floor(Int64,x[end]*10))


j = 1
for i in 2:size(state2)[2]
    if i < x[j]*10
        state2[:,i] = state2[:,i-1]
    elseif i > x[end]*10
        print("error with size of state2")
    else
        k=x[j]
        while k*10 < i
            global j
            j += 1
            k = x[j]
        end
        state2[:,i] = z[:,j]
    end
end

interval = interval*10
interval = [ceil.(m) for m in interval]


temp_2 = ones(1, size(state2)[1])
temp = temp_2*state2
save("data.jld", "data", temp)

#fix intervals starting at 0 instead of 1
for (i1, i2) in enumerate(interval)
    for (j1, j2) in enumerate(i2)
        if trunc(j2) == 0
            interval[i1][j1] = 1.0
        end
    end
end
save("interval.jld", "interval", interval[2:end-1])
