### Style guide
- Don't use `setting` directly in function body, use it as a default argument instead so that people can change it without modifying the function implement if they want to.
- When building a dataframe, try to make it confine to the "[tidy](https://cran.r-project.org/web/packages/tidyr/vignettes/tidy-data.html)" format, i.e. one element in each cell. It will make plotting and analysis much easier later on, as both pandas and tidyverse are built with this format in mind. Specifically, do not store column of different lengths in the same dataframe, as it will create nan when the shorter column is extracted. If you want to mark an event in time, use a boolean to indicate the event instead. If you have to use column of different lenghts
- Write comment for each function you create. Even a one-liner is better than nothing.
- If you are using a clever trick/hack in your code, explain in your comment. Indicate anything unusual that you do that needs mentioning
- Consider maintainability of you code. A hack you quickly scramble now to save you a few minute may cost you hours down the road when unexpected bugs appear out of nowhere. 