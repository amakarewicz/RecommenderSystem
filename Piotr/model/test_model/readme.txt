changed model for testing purposes
> doesnt't remove duplicates
> provide user_articles
> max limit of articles is amount of user's read articles

testing output data:

model        | user            | number_of_recomm | user articles | precision | recall    | model_ev
--------------------------------------------------------------------------------------------------------
popularity   | 0 (not in base) | 1 to 20          | number of     | from      | from      | from
author_p     | 1 to 1000       | lower than 20 if | read          | 0 to 1    | 0 to 1    | 1 to ... 
department_p | ...             | it's models max  | articles      | or NaN    | or NaN    |


example:
popularity, 8, 10, 34, 0.3, 0.06122, 1