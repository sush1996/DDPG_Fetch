# DDPG_Fetch
Exploring the performance of Prioritized Experience Replay (PER) with the DDPG+HER scheme on the Fetch Robotics Environemnt

Plots for Mean Success Rates for different Fetch Environments

<p float="middle">
  <img src="https://github.com/sush1996/DDPG_Fetch/blob/master/all_plots_fr.png?raw=true" width="400" />
  <img src="https://github.com/sush1996/DDPG_Fetch/blob/master/all_plots.png?raw=true" width="400" /> 
 
</p>

<p float="middle">
  <img src="https://github.com/sush1996/DDPG_Fetch/blob/master/all_plots_fp.png?raw=true" width="400" />
  <img src="https://github.com/sush1996/DDPG_Fetch/blob/master/all_plots_fs.png?raw=true" width="400" />
</p>

Performance Plots when varying the alpha parameter on PER
<p float="middle">
  <img src="https://github.com/sush1996/DDPG_Fetch/blob/master/alpha_plots_fp.png?raw=true" width="400" />
  <img src="https://github.com/sush1996/DDPG_Fetch/blob/master/alpha_plots_fs.png?raw=true" width="400" />
</p>

Addition of PER along with finetuning the alpha parameter boosts its performance. 

The inclusion of the PER algo within the DDPG-HER framework can be done in many ways, it could give greater performance boosts if combined well.
