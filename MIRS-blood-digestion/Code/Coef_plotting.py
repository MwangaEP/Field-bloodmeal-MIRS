# # summarizing logistic regression coefficients

# sss_coef_2 = lr_coef_df
# sss_coef_2.dropna(axis = 1, inplace = True)
# sss_coef_2["coef mean"] = sss_coef_2.mean(axis=1)
# sss_coef_2["coef sem"] = sss_coef_2.sem(axis=1)
# # sss_coef_2.to_csv("coef_repeatedCV_coef.csv")

# %%

# # plotting coefficients
# n_features = 25

# # sort the coefficients
# sss_coef_2.sort_values(
#     by = "coef mean", 
#     ascending = False, 
#     inplace = True
# )

# # select the top 25 and bottom 25 coefficients
# coef_plot_data = sss_coef_2.drop(
#     [
#         "coef sem", 
#         "coef mean"
#     ], 
#     axis = 1
# ).T

# coef_plot_data = coef_plot_data.iloc[:,:].drop(
#     coef_plot_data.columns[n_features:-n_features], 
#     axis = 1
# )


# # Plotting 

# plt.figure(figsize = (5,16))
# sns.barplot(
#                 data = coef_plot_data, 
#                 orient = "h", 
#                 palette = "plasma", 
#                 capsize = .2
#             )

# plt.ylabel("Wavenumbers", weight = "bold")
# plt.xlabel("Coefficients", weight = "bold")
# plt.savefig(os.path.join("..", "Results", "lgr_coef.png"), 
#             dpi = 500, 
#             bbox_inches = "tight"
#         )