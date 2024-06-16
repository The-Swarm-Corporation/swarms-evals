MMLU_PROMPT = """

**Prompt:**

1. **Understand the Problem:**
   - Read the problem statement carefully.
   - Identify what is being asked.
   - Highlight or note down key information and data.

2. **Plan Your Solution:**
   - Break down the problem into smaller, manageable parts.
   - Decide on the operations or steps needed to solve each part.

3. **Execute Step-by-Step:**
   - Solve each part one at a time.
   - Show all calculations and intermediate results clearly.
   - Verify each step before moving to the next.

4. **Combine Results:**
   - Bring together the results from each part.
   - Ensure all units and terms are consistent.
   - Perform any final calculations required.

5. **Double-Check Your Work:**
   - Review the entire solution.
   - Check for any mistakes or missed steps.
   - Confirm that the final answer addresses what was asked.

6. **Present the Answer:**
   - Clearly write the final answer.
   - Include units or labels as necessary.

---

**Example Problem:**

*If a car travels 60 miles in 1.5 hours, what is its average speed in miles per hour?*

1. **Understand the Problem:**
   - Distance traveled: 60 miles
   - Time taken: 1.5 hours
   - Question: Find the average speed (miles per hour).
   - Possible answers a ) 30 b) 40 c) 50 d) 60

2. **Plan Your Solution:**
   - To find the average speed, use the formula: speed = distance / time.

3. **Execute Step-by-Step:**
   - Distance = 60 miles
   - Time = 1.5 hours
   - Speed = 60 miles / 1.5 hours

4. **Combine Results:**
   - Speed = 60 / 1.5
   - Speed = 40 miles per hour

5. **Double-Check Your Work:**
   - Review calculation: 60 / 1.5 = 40
   - Units check: miles per hour

6. **Present the Answer:**
   - 40


"""
