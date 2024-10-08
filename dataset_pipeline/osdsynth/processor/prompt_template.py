# Distance
distance_template_questions = [
    "What is the distance between [A] and [B]?",
    "How far apart are [A] and [B]?",
    "How distant is [A] from [B]?",
    "How far is [A] from [B]?",
    "How close is [A] from [B]?",
    "Could you measure the distance between [A] and [B]?",
    "Can you tell me the distance of [A] from [B]?",
    "How far away is [A] from [B]?",
    "Can you provide the distance measurement between [A] and [B]?",
    "Can you give me an estimation of the distance between [A] and [B]?",
    "Could you provide the distance between [A] and [B]?",
    "How much distance is there between [A] and [B]?",
    "Tell me the distance between [A] and [B].",
    "Give me the distance from [A] to [B].",
    "Measure the distance from [A] to [B].",
    "Measure the distance between [A] and [B].",
]

distance_template_answers = [
    "[X]",
    "[A] and [B] are [X] apart.",
    "[A] is [X] away from [B].",
    "A distance of [X] exists between [A] and [B].",
    "[A] is [X] from [B].",
    "[A] and [B] are [X] apart from each other.",
    "They are [X] apart.",
    "The distance of [A] from [B] is [X].",
]

# Predicates
left_predicate_questions = [
    "Is [A] to the left of [B] from the viewer's perspective?",
    "Does [A] appear on the left side of [B]?",
    "Can you confirm if [A] is positioned to the left of [B]?",
]

left_true_responses = [
    "Yes, [A] is to the left of [B].",
    "Indeed, [A] is positioned on the left side of [B].",
    "Correct, you'll find [A] to the left of [B].",
]

left_false_responses = [
    "No, [A] is not to the left of [B].",
    "In fact, [A] is to the right of [B].",
    "Incorrect, [A] is not on the left side of [B].",
]

right_predicate_questions = [
    "Is [A] to the right of [B] from the viewer's perspective?",
    "Does [A] appear on the right side of [B]?",
    "Can you confirm if [A] is positioned to the right of [B]?",
]

right_true_responses = [
    "Yes, [A] is to the right of [B].",
    "Indeed, [A] is positioned on the right side of [B].",
    "Correct, you'll find [A] to the right of [B].",
]

right_false_responses = [
    "No, [A] is not to the right of [B].",
    "In fact, [A] is to the left of [B].",
    "Incorrect, [A] is not on the right side of [B].",
]

horizontal_aligned_responses = [
    "[A] and [B] are horizontally aligned without one being to the left or right of the other.",
    "[A] and [B] exist on the same horizontal plane, with neither left nor right distinction.",
    "Both [A] and [B] occupy the same horizontal space, without a left or right relationship.",
    "At the same horizontal level, [A] and [B] are aligned without implying one is to the left or right.",
    "[A] and [B] stand at equal distances, with no indication of one being to the left or right.",
    "No horizontal hierarchy exists between [A] and [B]; they are at a shared horizontal position.",
    "Occupying the same level, [A] and [B] show no horizontal order of left or right.",
    "[A] and [B] are aligned in such a way that negates a left or right relationship.",
    "Without one to the left or right of the other, [A] and [B] are horizontally aligned.",
    "[A] and [B] are at a level where distinguishing between left and right does not apply.",
]

above_predicate_questions = [
    "Is [A] above [B]?",
    "Does [A] appear over [B]?",
    "Can you confirm if [A] is positioned above [B]?",
]

above_true_responses = [
    "Yes, [A] is above [B].",
    "Indeed, [A] is positioned over [B].",
    "Correct, [A] is located above [B].",
]

above_false_responses = [
    "No, [A] is not above [B].",
    "Actually, [A] is below [B].",
    "Incorrect, [A] is not positioned above [B].",
]

vertical_aligned_responses = [
    "[A] and [B] share the same elevation without one being over the other.",
    "[A] and [B] exist on the same vertical plane, with neither above nor below.",
    "Both [A] and [B] occupy the same vertical space, without a top or bottom distinction.",
    "At the same elevation, [A] and [B] are aligned without implying one is atop the other.",
    "[A] and [B] stand at equal heights, with no indication of one being higher or lower.",
    "No vertical hierarchy exists between [A] and [B]; they are at a shared elevation.",
    "Occupying the same level, [A] and [B] show no vertical order of above or below.",
    "[A] and [B] are aligned in such a way that negates a top or bottom relationship.",
    "Without one over the other, [A] and [B]'s elevation is equivalent.",
    "[A] and [B] are at a level where distinguishing between higher and lower does not apply.",
]

below_predicate_questions = [
    "Is [A] below [B]?",
    "Does [A] appear under [B]?",
    "Can you confirm if [A] is positioned below [B]?",
]

below_true_responses = [
    "Yes, [A] is below [B].",
    "Indeed, [A] is positioned under [B].",
    "Correct, [A] is located below [B].",
]

below_false_responses = [
    "No, [A] is not below [B].",
    "Actually, [A] is above [B].",
    "Incorrect, [A] is not positioned below [B].",
]

wide_predicate_questions = [
    "Is [A] wider than [B]?",
    "Does [A] have a greater width compared to [B]?",
    "Can you confirm if [A] is wider than [B]?",
]

wide_true_responses = [
    "Yes, [A] is wider than [B].",
    "Indeed, [A] has a greater width compared to [B].",
    "Correct, the width of [A] exceeds that of [B].",
]

wide_false_responses = [
    "No, [A] is not wider than [B].",
    "In fact, [A] might be narrower than [B].",
    "Incorrect, [A]'s width does not surpass [B]'s.",
]

big_predicate_questions = [
    "Is [A] bigger than [B]?",
    "Does [A] have a larger size compared to [B]?",
    "Can you confirm if [A] is bigger than [B]?",
]

big_true_responses = [
    "Yes, [A] is bigger than [B].",
    "Indeed, [A] has a larger size compared to [B].",
    "Correct, [A] is larger in size than [B].",
]

big_false_responses = [
    "No, [A] is not bigger than [B].",
    "Actually, [A] might be smaller than [B].",
    "Incorrect, [A] is not larger than [B].",
]

tall_predicate_questions = [
    "Is [A] taller than [B]?",
    "Does [A] have a greater height compared to [B]?",
    "Can you confirm if [A] is taller than [B]?",
]

tall_true_responses = [
    "Yes, [A] is taller than [B].",
    "Indeed, [A] has a greater height compared to [B].",
    "Correct, [A] is much taller as [B].",
]

tall_false_responses = [
    "No, [A] is not taller than [B].",
    "In fact, [A] may be shorter than [B].",
    "Incorrect, [A]'s height is not larger of [B]'s.",
]

short_predicate_questions = [
    "Is [A] shorter than [B]?",
    "Does [A] have a lesser height compared to [B]?",
    "Can you confirm if [A] is shorter than [B]?",
]

short_true_responses = [
    "Yes, [A] is shorter than [B].",
    "Indeed, [A] has a lesser height compared to [B].",
    "Correct, [A] is not as tall as [B].",
]

short_false_responses = [
    "No, [A] is not shorter than [B].",
    "In fact, [A] may be taller than [B].",
    "Incorrect, [A]'s height does not fall short of [B]'s.",
]

thin_predicate_questions = [
    "Is [A] thinner than [B]?",
    "Does [A] have a lesser width compared to [B]?",
    "Can you confirm if [A] is thinner than [B]?",
]

thin_true_responses = [
    "Yes, [A] is thinner than [B].",
    "Indeed, [A] has a lesser width compared to [B].",
    "Correct, [A]'s width is less than [B]'s.",
]

thin_false_responses = [
    "No, [A] is not thinner than [B].",
    "In fact, [A] might be wider than [B].",
    "Incorrect, [A]'s width is not less than [B]'s.",
]

small_predicate_questions = [
    "Is [A] smaller than [B]?",
    "Does [A] have a smaller size compared to [B]?",
    "Can you confirm if [A] is smaller than [B]?",
]

small_true_responses = [
    "Yes, [A] is smaller than [B].",
    "Indeed, [A] has a smaller size compared to [B].",
    "Correct, [A] occupies less space than [B].",
]

small_false_responses = [
    "No, [A] is not smaller than [B].",
    "Actually, [A] might be larger than [B].",
    "Incorrect, [A] is not smaller in size than [B].",
]

behind_predicate_questions = [
    "Is [A] behind [B]?",
    "Is the position of [A] more distant than that of [B]?",
    "Does [A] lie behind [B]?",
    "Is [A] positioned behind [B]?",
    "Is [A] further to camera compared to [B]?",
    "Does [A] come behind [B]?",
    "Is [A] positioned at the back of [B]?",
    "Is [A] further to the viewer compared to [B]?",
]

behind_true = [
    "Yes.",
    "Yes, it is.",
    "Yes, it is behind [B].",
    "That is True.",
    "Yes, [A] is further from the viewer.",
    "Yes, [A] is behind [B].",
]

behind_false = [
    "No.",
    "No, it is not.",
    "No, it is in front of [B].",
    "That is False.",
    "No, [A] is closer to the viewer.",
    "No, [B] is in front of [A].",
]

front_predicate_questions = [
    "Is [A] in front of [B]?",
    "Is the position of [A] less distant than that of [B]?",
    "Does [A] lie in front of [B]?",
    "Is [A] positioned in front of [B]?",
    "Is [A] closer to camera compared to [B]?",
    "Does [A] come in front of [B]?",
    "Is [A] positioned before [B]?",
    "Is [A] closer to the viewer compared to [B]?",
]

front_true = [
    "Yes.",
    "Yes, it is.",
    "Yes, it is in front of [B].",
    "That is True.",
    "Yes, [A] is closer to the viewer.",
    "Yes, [A] is in front of [B].",
]

front_false = [
    "No.",
    "No, it is not.",
    "No, it is behind [B].",
    "That is False.",
    "No, [A] is further to the viewer.",
    "No, [A] is behind [B].",
]


# Choice
left_choice_questions = [
    "Which is more to the left, [A] or [B]?",
    "Between [A] and [B], which one appears on the left side from the viewer's perspective?",
    "Who is positioned more to the left, [A] or [B]?",
]

left_choice_responses = [
    "[X] is more to the left.",
    "From the viewer's perspective, [X] appears more on the left side.",
    "Positioned to the left is [X].",
]

right_choice_questions = [
    "Which is more to the right, [A] or [B]?",
    "Between [A] and [B], which one appears on the right side from the viewer's perspective?",
    "Who is positioned more to the right, [A] or [B]?",
]

right_choice_responses = [
    "[X] is more to the right.",
    "From the viewer's perspective, [X] appears more on the right side.",
    "Positioned to the right is [X].",
]

above_choice_questions = [
    "Which is above, [A] or [B]?",
    "Between [A] and [B], which one is positioned higher?",
    "Who is higher up, [A] or [B]?",
]

above_choice_responses = [
    "[X] is above.",
    "Positioned higher is [X].",
    "[X] is higher up.",
]

below_choice_questions = [
    "Which is below, [A] or [B]?",
    "Between [A] and [B], which one is positioned lower?",
    "Who is lower down, [A] or [B]?",
]

below_choice_responses = [
    "[X] is below.",
    "Positioned lower is [X].",
    "[X] is lower down.",
]

front_choice_questions = [
    "Which is in front, [A] or [B]?",
    "Between [A] and [B], which one is positioned in front?",
    "Who is more forward, [A] or [B]?",
]

front_choice_responses = [
    "[X] is in front.",
    "Positioned in front is [X].",
    "[X] is more forward.",
]

behind_choice_questions = [
    "Which is behind, [A] or [B]?",
    "Between [A] and [B], which one is positioned behind?",
    "Who is more backward, [A] or [B]?",
]

behind_choice_responses = [
    "[X] is behind.",
    "Positioned behind is [X].",
    "[X] is more backward.",
]

tall_choice_questions = [
    "Who is taller, [A] or [B]?",
    "Between [A] and [B], which one has more height?",
    "Which of these two, [A] or [B], stands taller?",
]

tall_choice_responses = [
    "[X] is taller.",
    "With more height is [X].",
    "Standing taller between the two is [X].",
]

short_choice_questions = [
    "Who is shorter, [A] or [B]?",
    "Between [A] and [B], which one has less height?",
    "Which of these two, [A] or [B], stands shorter?",
]

short_choice_responses = [
    "[X] is shorter.",
    "With less height is [X].",
    "Standing shorter between the two is [X].",
]

# Direction
direction_questions = ["If you are at [A], where will you find [B]?"]

direction_responses = ["[B] is roughly at [X] o'clock from [A].", "[A] will find [B] around the [X] o'clock direction."]


# Vertical and horizonal distance
vertical_distance_questions = [
    "What is the vertical distance between [A] and [B]?",
    "How far apart are [A] and [B] vertically?",
    "How distant is [A] from [B] vertically?",
    "How far is [A] from [B] vertically?",
    "Could you measure the vertical distance between [A] and [B]?",
    "Can you tell me the vertical distance between [A] and [B]?",
    "How far away is [A] from [B] vertically?",
    "Estimate the vertical distance between [A] and [B].",
    "Could you provide the vertical distance between [A] and [B]?",
    "How much distance is there between [A] and [B] vertically?",
    "Tell me the distance between [A] and [B] vertically.",
    "Give me the vertical distance from [A] to [B].",
    "Measure the vertical distance from [A] to [B].",
    "Measure the distance between [A] and [B] vertically.",
]

vertical_distance_answers = [
    "[X]",
    "[A] and [B] are [X] apart vertically.",
    "[A] is [X] away from [B] vertically.",
    "A vertical distance of [X] exists between [A] and [B].",
    "[A] is [X] from [B] vertically.",
    "[A] and [B] are [X] apart vertically from each other.",
    "Vertically, They are [X] apart.",
    "The vertical distance of [A] from [B] is [X].",
    "They are [X] apart.",
    "It is approximately [X].",
]

vertical_distance_supporting_answers = [
    "[bottom] directly supports [top], with no vertical distance between them.",
    "There is no vertical gap, as [bottom] is directly beneath and supporting [top].",
    "[top] is directly supported by [bottom], indicating zero vertical separation.",
    "With [bottom] supporting [top], they are in immediate vertical contact.",
    "[bottom] and [top] are connected vertically with [bottom] providing support directly below [top].",
    "The vertical support between [bottom] and [top] is direct, leaving no space.",
    "[bottom] directly below [top] serves as a supporting structure with no vertical distance.",
    "[top] receives direct support from [bottom] without any vertical gap.",
    "In their vertical arrangement, [bottom] supports [top] immediately, with no distance apart.",
    "Vertically, [bottom] is the immediate supporter of [top], with no discernible distance between them.",
]

vertical_distance_overlapping_answers = [
    "[A] and [B] overlap, so it's hard to tell how far apart they are vertically.",
    "Because [A] and [B] overlap vertically, figuring out the distance is hard.",
    "The overlap between [A] and [B] makes the vertical distance unclear.",
    "[A] and [B] are overlapping, which makes knowing the vertical distance difficult.",
    "It's hard to say the vertical distance since [A] and [B] overlap.",
    "Since [A] and [B] overlap, the vertical distance between them is hard to tell.",
    "[A] and [B]'s vertical overlap leaves the distance between them uncertain.",
    "Overlapping [A] and [B] makes it challenging to determine their vertical distance.",
]

horizontal_distance_questions = [
    "What is the horizontal distance between [A] and [B]?",
    "How far apart are [A] and [B] horizontally?",
    "How distant is [A] from [B] horizontally?",
    "How far is [A] from [B] horizontally?",
    "Could you measure the horizontal distance between [A] and [B]?",
    "Can you tell me the horizontal distance of [A] from [B]?",
    "Can you give me an estimation of the horizontal distance between [A] and [B]?",
    "Could you provide the horizontal distance between [A] and [B]?",
    "How much distance is there between [A] and [B] horizontally?",
    "Tell me the distance between [A] and [B] horizontally.",
    "Give me the horizontal distance from [A] to [B].",
    "Horizontal gap between [A] and [B].",
    "Measure the horizontal distance from [A] to [B].",
    "Measure the distance between [A] and [B] horizontally.",
]

horizontal_distance_answers = [
    "[X]",
    "[A] and [B] are [X] apart horizontally.",
    "[A] is [X] away from [B] horizontally.",
    "A horizontal distance of [X] exists between [A] and [B].",
    "[A] is [X] from [B] horizontally.",
    "[A] and [B] are [X] apart horizontally from each other.",
    "Horizontally, They are [X] apart.",
    "The horizontal distance of [A] from [B] is [X].",
    "They are [X] apart.",
    "It is approximately [X].",
]

# Width/Height
width_questions = [
    "Measure the width of [A].",
    "Determine the horizontal dimensions of [A].",
    "Find out how wide [A] is.",
    "What is the width of [A]?",
    "How wide is [A]?",
    "What are the dimensions of [A] in terms of width?",
    "Could you tell me the horizontal size of [A]?",
    "What is the approximate width of [A]?",
    "How wide is [A]?",
    "How much space does [A] occupy horizontally?",
    "How big is [A] in terms of width?",
    "What is the radius of [A]?",
]
width_answers = [
    "[X]",
    "The width of [A] is [X].",
    "[A] is [X] wide.",
    "[A] is [X] in width.",
    "It is [X].",
]

height_questions = [
    "Measure the height of [A].",
    "Determine the vertical dimensions of [A].",
    "Find out how tall [A] is.",
    "What is the height of [A]?",
    "How tall is [A]?",
    "What are the dimensions of [A] in terms of height?",
    "Could you tell me the vericall size of [A]?",
    "What is the approximate height of [A]?",
    "How tall is [A]?",
    "How much space does [A] occupy vertically?",
    "How tall is [A]?",
    "How tall is [A] in terms of height?",
]
height_answers = [
    "[X]",
    "The height of [A] is [X].",
    "[A] is [X] tall.",
    "[A] is [X] in height.",
    "It is [X].",
]
