-- Here I am Creating DB to Store thr Excel files
CREATE DATABASE HR_Sale;
-- To start to use DB;
USE hr_sale;
-- to See al the table sin the DB 
SHOW TABLES;
-- To View specified tabel in that DB
DESCRIBE `human resources`;
-- or
SHOW COLUMNS FROM `human resources`;

-- to rename the Table name in the DB
RENAME TABLE `human resources` TO hr;

-- to recreate or duplicate the table
CREATE TABLE hr_1 AS
SELECT * FROM hr; 

SHOW INDEX FROM hr_1;
---------------------------------------------------
-- Projext Work----------------------------------
SELECT * FROM hr;
-- USED to RENMAE COlumn nmae in table
ALTER TABLE hr
CHANGE COLUMN ï»¿id emp_id VARCHAR(20) NULL;

SELECT * FROM hr;

-- beloew code used to run the update code in safe more it turn on the safe mode
SET sql_safe_updates = 0;

-- here we are standardatising the Date columns
UPDATE hr
SET birthdate = CASE 
	WHEN birthdate LIKE '%/%' THEN date_format(str_to_date(birthdate,'%m/%d/%Y'),'%Y-%m-%d')
    WHEN birthdate LIKE '%-%' THEN date_format(str_to_date(birthdate,'%m-%d-%Y'),'%Y-%m-%d')
    ELSE NULL 
END;

ALTER TABLE hr
MODIFy COLUMN birthdate DATE;

UPDATE hr
SET hire_date = CASE 
	WHEN hire_date LIKE '%/%' THEN date_format(str_to_date(hire_date,'%m/%d/%Y'),'%Y-%m-%d')
    WHEN hire_date LIKE '%-%' THEN date_format(str_to_date(hire_date,'%m-%d-%Y'),'%Y-%m-%d')
    ELSE NULL 
END;

ALTER TABLE hr
MODIFy COLUMN hire_date DATE;

SELECT termdate FROM hr ; 

UPDATE hr
SET termdate = DATE(str_to_date(termdate,'%Y-%m-%d %H:%i:%s UTC'))
WHERE termdate IS NOT NULL AND termdate !=''; 

UPDATE hr
SET termdate = NULL
WHERE termdate = '';


ALTER TABLE hr
MODIFY COLUMN termdate DATE; 

SELECT termdate FROM hr;

SHOW COLUMNS FROM hr LIKE 'termdate';

-- beloow code used for calculate age 
ALTER TABLE hr ADD COLUMN age INT;

UPDATE hr 
SET age = timestampdiff(YEAR,birthdate,CURDATE());

SELECT birthdate, age FROM hr;

-- Chekcing outliers;


SELECT 
	MIN(age) AS low_age,
    MAX(age) AS high_age
FROM hr;

SELECT termdate FROM hr;

-- 1. What is the gender breakdown of employees in the company?;
SELECT gender, Count(*) AS COunt 
FROM hr 
WHERE age >=18 AND termdate IS NULL
GROUP BY gender;

-- 2. What is the race/ethnicity breakdown of employees in the company?
SELECT race, Count(*) AS count
FROM hr 
WHERE age >=18 AND termdate IS NULL
GROUP BY race
ORDER BY count(*) DESC;
-- 3. What is the age distribution of employees in the company?
SELECT 
CASE
	WHEN age >=18 AND age <=24 THEN '18-24'
    WHEN age >=25 AND age <=34 THEN '25-34'
    WHEN age >=35 AND age <=44 THEN '35-44'
    WHEN age >=45 AND age <=54 THEN '45-54'
    WHEN age >=55 AND age <=64 THEN '55-64'
	ELSE '65+'
END AS age_dis,gender,
count(*) AS Count
FROM hr
WHERE age>=18 AND termdate IS NULL
GROUP BY age_dis,gender
ORDER BY age_dis, gender;

-- 4. How many employees work at headquarters versus remote locations?
SELECT location, count(*) AS Count
FROM hr
WHERE age>=18 AND termdate IS NULL
GROUP BY location 
ORDER BY count(*);

-- 5. What is the average length of employment for employees who have been terminated?
SELECT 
ROUND(AVG(DATEDIFF(termdate,hire_date)/365),0) AS len_date
FROM hr
WHERE termdate IS NOT NULL AND termdate <= current_date();

-- 6. How does the gender distribution vary across departments and job titles?
SELECT department, gender, count(*) AS gen_des_dep_job
FROM hr
WHERE age>=18 AND termdate IS NULL
GROUP BY gender, department
ORDER BY department, gender;
-- 7. What is the distribution of job titles across the company?
SELECT jobtitle, count(*) AS Count
FROM hr
WHERE age>=18 AND termdate IS NULL
GROUP BY jobtitle
ORDER BY count(*) DESC;

-- 8. Which department has the highest turnover rate?
SELECT 
department, 
total_count, 
terminated_count,
(terminated_count/total_count) AS terminated_rate
FROM (
	SELECT 
	department,
	count(*) AS total_count,
    SUM(CASE WHEN termdate IS NOT NULL AND termdate <= curdate() THEN 1 ELSE 0 END) AS terminated_count
    FROM hr
    WHERE age >=18
    GROUP BY department) AS subquery
    ORDER BY terminated_rate DESC;
    -- 9. What is the distribution of employees across locations by city and state?
    SELECT location_state, count(*)
    FROM hr
    WHERE age >=18 AND termdate IS NULL
    GROUP BY location_state
    ORDER BY count(*) DESC;
    -- 10. How has the company's employee count changed over time based on hire and term dates?
SELECT 
year,
hire,
termination,
hire-termination AS net_change,
ROUND((((hire-termination)/hire)*100),2) AS net_change_percent
FROM ( 
SELECT 
YEAR(hire_date) AS year
, count(*) AS hire,
SUM(CASE WHEN termdate IS NOT NULL AND termdate != curdate() THEN 1 ELSE 0 END ) AS termination
FROM hr
WHERE age >=18
GROUP BY year) AS subquery
ORDER BY year;
-- 11. What is the tenure distribution for each department?
SELECT department AS termination,
ROUND(AVG(datediff(termdate,hire_date)/365),0) AS tenure_days,
count(*)
FROM hr
WHERE age >=18 AND termdate IS NOT NULL AND termdate <= curdate()
GROUP BY department


    
    
    
    









