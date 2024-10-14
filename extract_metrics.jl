module SolutionMetrics
using HallThruster
using JSON3

# Function to extract performance metrics from the solution object
function extract_performance_metrics(solution::HallThruster.Solution; save_every_n_grid_points::Int=10)
    times = solution.t  #(no interval for time)
    thrust_output = HallThruster.thrust(solution)
    discharge_current_output = HallThruster.discharge_current(solution)
    
    z_positions = solution.params.z_cell # spatial positions (z / L)
    z_normalized = z_positions ./ solution.params.L_ch  # Normalized positions

    # Save every nth grid point for spatial metrics (e.g., ion velocity)
    # ion_density = solution[:ni, 1][1:save_every_n_grid_points:end]
    # neutral_density = solution[:nn][1:save_every_n_grid_points:end]
    # potential_output = solution[:ϕ][1:save_every_n_grid_points:end]
    # electron_temperature = solution[:Tev][1:save_every_n_grid_points:end]
    ion_velocity = solution[:ui, 1][1:save_every_n_grid_points:end]
    # magnetic_field = solution[:B][1:save_every_n_grid_points:end]
    # current_density = solution[:ji][1:save_every_n_grid_points:end]

    result_data = Dict(
        "thrust" => thrust_output,
        "time" => times,  # Keep all time data intact
        "discharge_current" => discharge_current_output,
        "z_normalized" => z_normalized,   #[1:save_every_n_grid_points:end],  #Save every nth point
        # "ion_density" => ion_density,
        # "neutral_density" => neutral_density,
        # "potential" => potential_output,
        # "electron_temperature" => electron_temperature,
        "ion_velocity" => ion_velocity[1:save_every_n_grid_points:end],
        # "magnetic_field" => magnetic_field,
        # "current_density" => current_density
    )

    # Return as JSON string
    return JSON3.write(result_data)
end

# saving every 10th grid point
function extract_time_averaged_metrics(solution::HallThruster.Solution, n_save::Int; save_every_n_grid_points::Int=10)
    # Start time-averaging after 40% of the simulation
    average_start_frame = Int(0.4 * n_save)
    time_averaged_solution = HallThruster.time_average(solution, average_start_frame)
    
    # after averaging
    return extract_performance_metrics(time_averaged_solution, save_every_n_grid_points=save_every_n_grid_points)
end

end
