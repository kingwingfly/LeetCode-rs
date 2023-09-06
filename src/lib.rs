/// judge if two numbers in the Vec could sum to target
#[cfg(target_feature = "all")]
fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
    use std::collections::HashMap;

    let mut mp = HashMap::with_capacity(nums.len());

    for i in 0..nums.len() {
        match mp.get(&nums[i]) {
            Some(&t) => return vec![i as i32, t as i32],
            None => {
                mp.insert(target - nums[i], i);
            }
        }
    }
    unreachable!();
}

// #[cfg(target_feature = "all")]
// Definition for singly-linked list.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

#[cfg(target_feature = "all")]
impl ListNode {
    #[inline]
    fn new(val: i32) -> Self {
        ListNode { next: None, val }
    }
}

#[cfg(target_feature = "all")]
fn add_two_numbers(l1: Option<Box<ListNode>>, l2: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    fn carried(
        l1: Option<Box<ListNode>>,
        l2: Option<Box<ListNode>>,
        mut carry: i32,
    ) -> Option<Box<ListNode>> {
        if l1.is_none() && l2.is_none() && carry == 0 {
            return None;
        } else {
            return Some(Box::new(ListNode {
                next: {
                    carried(
                        l1.and_then(|x| {
                            carry += x.val;
                            x.next
                        }),
                        l2.and_then(|x| {
                            carry += x.val;
                            x.next
                        }),
                        carry / 10,
                    )
                },
                val: carry % 10,
            }));
        }
    }
    carried(l1, l2, 0)
}

#[cfg(target_feature = "all")]
fn length_of_longest_substring(s: String) -> i32 {
    let ptr0 = s.as_ptr();
    let mut ptr1 = s.as_ptr();
    let mut ptr2 = s.as_ptr();
    let mut v = [-1isize; 128];
    let mut ans = 0;
    for _ in 0..s.len() {
        unsafe {
            if v[*ptr2 as usize] >= 0 {
                ans = std::cmp::max(ans, ptr2.offset_from(ptr1));
                for i in ptr1.offset_from(ptr0)..=v[*ptr2 as usize] {
                    v[*ptr0.offset(i) as usize] = -1;
                    ptr1 = ptr1.add(1)
                }
            } else {
                ans = std::cmp::max(ans, ptr2.offset_from(ptr1) + 1);
            }
            v[*ptr2 as usize] = ptr2.offset_from(ptr0);
            ptr2 = ptr2.add(1);
        }
    }
    ans as i32
}

#[cfg(target_feature = "all")]
fn find_median_sorted_arrays(nums1: Vec<i32>, nums2: Vec<i32>) -> f64 {
    fn find_pos(nums1: &Vec<i32>, nums2: &Vec<i32>, mut pos: usize) -> f64 {
        let (mut l1, mut l2) = (0, 0);
        loop {
            let a = nums1
                .get(l1 + (pos >> 1))
                .or(nums1.get(l1.max(nums1.len() - 1)));
            let b = nums2
                .get(l2 + (pos >> 1))
                .or(nums2.get(l2.max(nums2.len() - 1)));
            match (a, b) {
                (None, Some(_)) => return nums2[l2 + pos] as f64,
                (Some(_), None) => return nums1[l1 + pos] as f64,
                (Some(a), Some(b)) => match a.cmp(b) {
                    std::cmp::Ordering::Less => {
                        if pos == 1 {
                            pos = 0;
                            l1 += 1;
                        } else {
                            let max_delta = nums1.len() - l1;
                            l1 += pos >> 1;
                            pos -= (pos >> 1).min(max_delta);
                        }
                    }
                    _ => {
                        if pos == 1 {
                            pos = 0;
                            l2 += 1;
                        } else {
                            let max_delta = nums2.len() - l2;
                            l2 += pos >> 1;
                            pos -= (pos >> 1).min(max_delta);
                        }
                    }
                },
                _ => unreachable!(),
            }
            if pos == 0 {
                match (nums1.get(l1), nums2.get(l2)) {
                    (None, Some(&b)) => return b as f64,
                    (Some(&a), None) => return a as f64,
                    (Some(&a), Some(&b)) => return a.min(b) as f64,
                    _ => unreachable!(),
                }
            }
        }
    }
    match ((nums1.len() + nums2.len()) & 1) {
        0 => {
            let pos1 = (nums1.len() + nums2.len()) >> 1;
            let pos2 = pos1 - 1;
            (find_pos(&nums1, &nums2, pos1) + find_pos(&nums1, &nums2, pos2)) / 2.
        }
        1 => {
            let pos = (nums1.len() + nums2.len()) >> 1;
            find_pos(&nums1, &nums2, pos)
        }
        _ => unreachable!(),
    }
}

#[cfg(target_feature = "all")]
fn longest_palindrome(s: String) -> String {
    fn foo(s: &[u8], mut l: isize, mut r: usize) -> (usize, usize) {
        while l >= 0 && r < s.len() && s[l as usize] == s[r] {
            l -= 1;
            r += 1
        }
        ((l + 1) as usize, r - 1)
    }
    let (mut start, mut end) = (0, 0);
    for i in 0..s.len() {
        let (l, r) = foo(s.as_bytes(), i as isize, i);
        if r - l > end - start {
            (start, end) = (l, r);
        }
    }
    for i in 0..s.len() - 1 {
        let (l, r) = foo(s.as_bytes(), i as isize, i + 1);
        if r >= l && r - l > end - start {
            (start, end) = (l, r);
        }
    }
    s[start..=end].to_string()
}

#[cfg(target_feature = "all")]
fn convert1(s: String, num_rows: i32) -> String {
    let mut ans = String::new();
    for mut row in 0..num_rows {
        let space1 = if row == 0 || row == num_rows - 1 {
            (num_rows << 1) - 2
        } else {
            (num_rows - 1 - row) << 1
        };
        let space2 = if row == 0 || row == num_rows - 1 {
            space1
        } else {
            row << 1
        };
        let mut i = 0;
        loop {
            if row as usize >= s.len() {
                break;
            }
            ans.push(s.as_bytes()[row as usize] as char);
            if i & 1 == 0 {
                row += space1;
            } else {
                row += space2;
            }
            i += 1;
        }
    }
    ans
}

#[cfg(target_feature = "all")]
fn convert(s: String, num_rows: i32) -> String {
    if num_rows <= 1 {
        return s;
    }
    let (mut i, mut flag) = (0, -1);
    let mut v = vec![String::new(); num_rows as usize];
    for c in s.chars() {
        if i == 0 || i == num_rows - 1 {
            flag = -flag;
        }
        v[i as usize].push(c);
        i += flag;
    }
    v.concat()
}

#[cfg(target_feature = "all")]
fn reverse(mut x: i32) -> i32 {
    let mut ret = 0;
    while x >= 10 || x <= -10 {
        let last = x % 10;
        ret = ret * 10 + last;
        x /= 10;
    }
    let last = x;
    if ret > i32::MAX / 10 || ret < i32::MIN / 10 {
        return 0;
    }
    ret = ret * 10 + last;
    ret
}

#[cfg(target_feature = "all")]
fn my_atoi(s: String) -> i32 {
    let mut ret = 0;
    let mut flag = 0;
    let upper = i32::MAX / 10;
    let down = i32::MIN / 10;
    for c in s.chars() {
        if ((ret < down || ret > upper) && ('0'..='9').contains(&c))
            || (ret == upper && (c == '8' || c == '9'))
            || (ret == down && c == '9')
        {
            match flag {
                1 => return i32::MAX,
                -1 => return i32::MIN,
                _ => unreachable!(),
            }
        }
        match (c, flag) {
            ('-', 0) => flag = -1,
            ('+', 0) => flag = 1,
            ('0'..='9', 1) => ret = 10 * ret + c.to_digit(10).unwrap() as i32,
            ('0'..='9', -1) => ret = 10 * ret - c.to_digit(10).unwrap() as i32,
            ('0'..='9', 0) => {
                ret = c.to_digit(10).unwrap() as i32;
                flag = 1;
            }
            (' ', 0) => {}
            _ => return ret,
        }
    }
    ret
}

#[cfg(target_feature = "all")]
fn is_palindrome(mut x: i32) -> bool {
    if x < 0 {
        return false;
    }
    if x > 2147447412 {
        return false;
    }
    let mut length = {
        if x >= 1000000000 {
            10
        } else if x >= 100000000 {
            9
        } else if x >= 10000000 {
            8
        } else if x >= 1000000 {
            7
        } else if x >= 100000 {
            6
        } else if x >= 10000 {
            5
        } else if x >= 1000 {
            4
        } else if x >= 100 {
            3
        } else if x >= 10 {
            2
        } else {
            1
        }
    };
    for i in 1..=(length >> 1) {
        let mut right = x % 10i32.pow(i);
        right /= 10i32.pow(i - 1);
        let left = x / 10i32.pow(length - i);
        if (left - right) % 10 != 0 {
            return false;
        }
    }
    true
    // if (x < 0 || (x % 10 == 0 && x != 0)) {
    //     return false;
    // }

    // let mut temp = 0;
    // while (x > temp) {
    //     temp = temp * 10 + x % 10;
    //     x /= 10;
    // }

    // 当数字长度为奇数时，我们可以通过 temp/10 去除处于中位的数字。
    // 例如，当输入为 12321 时，在 while 循环的末尾我们可以得到 x = 12，temp = 123，
    // 由于处于中位的数字不影响回文（它总是与自己相等），所以我们可以简单地将其去除。
    // return x == temp || x == temp / 10;
}

#[cfg(feature = "all")]
fn is_match(s: String, p: String) -> bool {
    let s: Vec<char> = s.chars().collect(); // 使用vec胖指针索引 较 s.chars().nth() 快得多。
    let p: Vec<char> = p.chars().collect();
    let match_c = |i, j| -> bool {
        // match函数直接上闭包
        i != 0 && (p[j - 1] == '.' || s[i - 1] == p[j - 1]) //因为返回值是bool类型 用&&比if更省事
    };
    let mut dp = vec![vec![false; p.len() + 1]; s.len() + 1]; // 不能用数组 其初始化个数必须为常量
    dp[0][0] = true;
    (0..=s.len()).for_each(|i| {
        //迭代器较循环更易于优化
        (1..=p.len()).for_each(|j| {
            dp[i][j] = if p[j - 1] == '*' {
                match_c(i, j - 1) && dp[i - 1][j] || dp[i][j - 2]
            } else {
                match_c(i, j) && dp[i - 1][j - 1]
            };
        })
    });
    dp[s.len()][p.len()]
}

#[cfg(feature = "all")]
fn max_area(height: Vec<i32>) -> i32 {
    let (mut i, mut j) = (0, height.len() - 1);
    let cal = |i: usize, j: usize| -> usize { (j - i) * height[i].min(height[j]) as usize };
    let mut area = 0;
    while i < j {
        area = area.max(cal(i, j));
        match height[i].cmp(&height[j]) {
            std::cmp::Ordering::Less => i += 1,
            _ => j -= 1,
        }
    }
    area as i32
}

#[cfg(feature = "all")]
fn int_to_roman(mut num: i32) -> String {
    let mut ans = String::new();
    let foo = |i: i32, symbol: &str, num: &mut i32, ans: &mut String| loop {
        if *num >= i {
            *num = *num - i;
            ans.push_str(symbol);
        } else {
            break;
        }
    };
    for (i, symbol) in [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ] {
        foo(i, symbol, &mut num, &mut ans);
    }
    ans
}

#[cfg(feature = "all")]
pub fn roman_to_int(mut s: String) -> i32 {
    let mut ans = 0;
    let foo = |i: i32, symbol: &str, ans: &mut i32, s: &mut String| loop {
        if s.starts_with(symbol) {
            *ans += i;
            s.drain(..symbol.len());
        } else {
            break;
        }
    };
    for (i, symbol) in [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ] {
        foo(i, symbol, &mut ans, &mut s);
    }
    ans
}

#[cfg(feature = "all")]
fn longest_common_prefix(strs: Vec<String>) -> String {
    let mut ans = String::new();
    let mut strs_vec: Vec<_> = strs.iter().map(|s| s.chars()).collect();
    loop {
        match strs_vec[0].next() {
            None => return ans,
            Some(c1) => {
                for s in strs_vec[1..].iter_mut() {
                    match s.next() {
                        None => return ans,
                        Some(c2) => {
                            if c1 != c2 {
                                return ans;
                            }
                        }
                    }
                }
                ans.push(c1);
            }
        }
    }
    ans
}

#[cfg(feature = "all")]
fn three_sum(mut nums: Vec<i32>) -> Vec<Vec<i32>> {
    let mut ans = vec![];
    if nums.len() < 3 {
        return ans;
    }
    nums.sort();
    if nums[0] > 0 {
        return ans;
    }
    let loop_to_next = |x: &mut usize, step: isize, nums: &Vec<i32>, edge: usize| {
        *x = (*x as isize + step) as usize;
        if step > 0 {
            while *x < edge && nums[*x] == nums[(*x as isize - step) as usize] {
                *x = (*x as isize + step) as usize;
            }
        } else {
            while *x > edge && nums[*x] == nums[(*x as isize - step) as usize] {
                *x = (*x as isize + step) as usize;
            }
        }
    };
    let mut i = 0;
    while i < nums.len() - 2 && nums[i] <= 0 {
        let (mut l, mut r) = (i + 1, nums.len() - 1);
        while l < r {
            let total = nums[i] + nums[l] + nums[r];
            match total.cmp(&0) {
                std::cmp::Ordering::Less => loop_to_next(&mut l, 1, &nums, r),
                std::cmp::Ordering::Equal => {
                    ans.push(vec![nums[i], nums[l], nums[r]]);
                    loop_to_next(&mut l, 1, &nums, r);
                    loop_to_next(&mut r, -1, &nums, l);
                }
                std::cmp::Ordering::Greater => loop_to_next(&mut r, -1, &nums, l),
            }
        }
        loop_to_next(&mut i, 1, &nums, nums.len());
    }
    ans
}

#[cfg(feature = "all")]
fn three_sum_closest(mut nums: Vec<i32>, target: i32) -> i32 {
    assert!(nums.len() >= 3);
    let mut ans = i32::MAX;
    let mut delta = ans;
    nums.sort();
    let mut i = 0;
    let loop_to_next = |x: &mut usize, step: isize, nums: &Vec<i32>, edge: usize| {
        *x = (*x as isize + step) as usize;
        if step > 0 {
            while *x < edge && nums[*x] == nums[(*x as isize - step) as usize] {
                *x = (*x as isize + step) as usize;
            }
        } else {
            while *x > edge && nums[*x] == nums[(*x as isize - step) as usize] {
                *x = (*x as isize + step) as usize;
            }
        }
    };
    let refesh = |total: i32, target: &i32, delta: &mut i32, ans: &mut i32| {
        let new_dalta = (*target - total).abs();
        if *delta > new_dalta {
            *delta = new_dalta;
            *ans = total;
        }
    };
    while i < nums.len() - 2 {
        let (mut l, mut r) = (i + 1, nums.len() - 1);
        while l < r {
            let total = nums[i] + nums[l] + nums[r];
            match total.cmp(&target) {
                std::cmp::Ordering::Less => {
                    refesh(total, &target, &mut delta, &mut ans);
                    loop_to_next(&mut l, 1, &nums, r);
                }
                std::cmp::Ordering::Equal => return total,
                std::cmp::Ordering::Greater => {
                    refesh(total, &target, &mut delta, &mut ans);
                    loop_to_next(&mut r, -1, &nums, l)
                }
            }
        }
        loop_to_next(&mut i, 1, &nums, nums.len());
    }
    ans
}

#[cfg(feature = "all")]
macro_rules! hashmap {
    ($($key:expr => $value: expr), *) => {
        {
            let mut map = std::collections::HashMap::new();
            $(map.insert($key, $value);)*
            map
        }
    };
}

#[cfg(feature = "all")]
struct LetterCombination<'a> {
    digits: Vec<&'a str>,
    ans: Vec<String>,
}

#[cfg(feature = "all")]
impl<'a> LetterCombination<'a> {
    fn new(digits: String) -> Self {
        assert!(!digits.contains("1"));
        let dash = ["abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"];
        Self {
            digits: digits.bytes().map(|i| dash[(i - b'2') as usize]).collect(),
            ans: vec![],
        }
    }

    fn dfs(&mut self) -> Vec<String> {
        fn recursive(lc: &mut LetterCombination, prefix: String, depth: usize) {
            if depth == lc.digits.len() && prefix.len() > 0 {
                lc.ans.push(prefix);
            } else if depth != lc.digits.len() {
                for c in lc.digits[depth].chars() {
                    recursive(lc, format!("{}{}", prefix, c), depth + 1)
                }
            }
        }
        recursive(self, String::new(), 0);
        std::mem::take(&mut self.ans)
    }
}

#[cfg(feature = "all")]
fn letter_combinations(digits: String) -> Vec<String> {
    let mut lc = LetterCombination::new(digits);
    lc.dfs()
}

#[cfg(feature = "all")]
fn four_sum(mut nums: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
    let mut ans = vec![];
    if nums.len() < 4 {
        return ans;
    }
    let (mut i, mut j) = (0, nums.len() - 1);
    nums.sort();
    let loop_to_next = |x: &mut usize, step: isize, nums: &Vec<i32>, edge: usize| {
        *x = (*x as isize + step) as usize;
        if step > 0 {
            while *x < edge && nums[*x] == nums[(*x as isize - step) as usize] {
                *x = (*x as isize + step) as usize;
            }
        } else {
            while *x > edge && nums[*x] == nums[(*x as isize - step) as usize] {
                *x = (*x as isize + step) as usize;
            }
        }
    };
    enum MyOption {
        Some(i32),
        Over,
        Below,
    }
    let sum_nums = |index: [usize; 4], nums: &Vec<i32>| -> MyOption {
        let mut total = 0;
        for i in index.into_iter() {
            if (nums[i] >= 0 && i32::MAX - nums[i] >= total)
                || (nums[i] < 0 && total >= i32::MIN - nums[i])
            {
                total += nums[i];
            } else if nums[i] >= 0 {
                return MyOption::Over;
            } else {
                return MyOption::Below;
            }
        }
        MyOption::Some(total)
    };
    while i < j && i + 3 < nums.len() {
        while i < j && i + 2 < j && j >= 3 {
            let (mut l, mut r) = (i + 1, j - 1);
            while l < r {
                let total = match sum_nums([i, l, r, j], &nums) {
                    MyOption::Some(total) => total,
                    MyOption::Over => {
                        loop_to_next(&mut r, -1, &nums, l);
                        continue;
                    }
                    MyOption::Below => {
                        loop_to_next(&mut l, 1, &nums, r);
                        continue;
                    }
                };
                match total.cmp(&target) {
                    std::cmp::Ordering::Less => loop_to_next(&mut l, 1, &nums, r),
                    std::cmp::Ordering::Equal => {
                        ans.push([i, l, r, j].map(|x| nums[x]).into_iter().collect());
                        loop_to_next(&mut l, 1, &nums, r);
                        loop_to_next(&mut r, -1, &nums, l);
                    }
                    std::cmp::Ordering::Greater => loop_to_next(&mut r, -1, &nums, l),
                }
            }
            loop_to_next(&mut i, 1, &nums, j);
        }
        i = 0;
        loop_to_next(&mut j, -1, &nums, 0);
    }
    ans
}

#[cfg(feature = "all")]
// Definition for singly-linked list.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

#[cfg(feature = "all")]
fn remove_nth_from_end(head: Option<Box<ListNode>>, n: i32) -> Option<Box<ListNode>> {
    let mut dummy = ListNode { val: 0, next: head };
    unsafe {
        let mut fast = &mut dummy as *mut ListNode;
        let mut slow = fast;
        for _ in 0..n {
            fast = (*fast).next.as_mut()?.as_mut();
        }
        while (*fast).next.is_some() {
            fast = (*fast).next.as_mut()?.as_mut();
            slow = (*slow).next.as_mut()?.as_mut();
        }
        (*slow).next = (*slow).next.take()?.next;
    }
    dummy.next
}

#[cfg(feature = "all")]
fn is_valid(s: String) -> bool {
    let mut stack = vec![];
    for c in s.chars() {
        match c {
            '(' | '[' | '{' => {
                stack.push(c);
                continue;
            }
            ')' => {
                if Some('(') == stack.pop() {
                    continue;
                }
            }
            ']' => {
                if Some('[') == stack.pop() {
                    continue;
                }
            }
            '}' => {
                if Some('{') == stack.pop() {
                    continue;
                }
            }
            _ => unreachable!(),
        }
        return false;
    }
    stack.is_empty()
}

#[cfg(feature = "all")]
fn merge_two_lists(
    mut list1: Option<Box<ListNode>>,
    mut list2: Option<Box<ListNode>>,
) -> Option<Box<ListNode>> {
    let mut head = None;
    let mut cur = &mut head;
    loop {
        match (list1, list2) {
            (Some(mut l1), Some(mut l2)) => {
                if l1.val < l2.val {
                    list1 = l1.next.take();
                    list2 = Some(l2);
                    cur = &mut cur.insert(l1).next
                } else {
                    list1 = Some(l1);
                    list2 = l2.next.take();
                    cur = &mut cur.insert(l2).next;
                }
            }
            (x, y) => {
                *cur = x.or(y);
                break;
            }
        }
    }
    head
}

#[cfg(feature = "all")]
fn generate_parenthesis(n: i32) -> Vec<String> {
    let mut ans = vec![];
    fn recursive(n: i32, ans: &mut Vec<String>, s: String, l_num: i32, r_num: i32, depth: i32) {
        if 2 * n == depth {
            ans.push(s);
        } else {
            for c in ['(', ')'] {
                if c == '(' && l_num < n {
                    recursive(n, ans, format!("{}{}", s, c), l_num + 1, r_num, depth + 1);
                } else if c == ')' && l_num > r_num {
                    recursive(n, ans, format!("{}{}", s, c), l_num, r_num + 1, depth + 1);
                }
            }
        }
    }
    recursive(n, &mut ans, String::new(), 0, 0, 0);
    ans
}

#[cfg(feature = "all")]
fn merge_k_lists(lists: Vec<Option<Box<ListNode>>>) -> Option<Box<ListNode>> {
    fn sort_two(
        mut list1: Option<Box<ListNode>>,
        mut list2: Option<Box<ListNode>>,
    ) -> Option<Box<ListNode>> {
        let mut node = None;
        let mut cur = &mut node;
        loop {
            match (list1, list2) {
                (Some(mut l1), Some(mut l2)) => {
                    if l1.val < l2.val {
                        list1 = l1.next.take();
                        list2 = Some(l2);
                        cur = &mut cur.insert(l1).next;
                    } else {
                        list1 = Some(l1);
                        list2 = l2.next.take();
                        cur = &mut cur.insert(l2).next;
                    }
                }
                (x, y) => {
                    *cur = x.or(y);
                    break;
                }
            }
        }
        node
    }
    fn binary_sort(mut lists: Vec<Option<Box<ListNode>>>) -> Option<Box<ListNode>> {
        if lists.len() <= 1 {
            lists.pop()?
        } else if lists.len() == 2 {
            sort_two(lists.pop().unwrap(), lists.pop().unwrap())
        } else {
            let right = lists.split_off(lists.len() >> 1);
            let left = binary_sort(lists);
            let right = binary_sort(right);
            binary_sort(vec![left, right])
        }
    }
    binary_sort(lists)
}

#[cfg(feature = "all")]
use std::cmp::{Ord, Ordering, PartialEq};
#[cfg(feature = "all")]
use std::collections::BinaryHeap;

#[cfg(feature = "all")]
impl Ord for ListNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // 默认是最大堆，这里颠倒顺序，实现最小堆。
        other.val.cmp(&self.val)
    }
}

#[cfg(feature = "all")]
impl PartialOrd for ListNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(feature = "all")]
fn merge_k_lists_heap(lists: Vec<Option<Box<ListNode>>>) -> Option<Box<ListNode>> {
    if lists.is_empty() {
        return None;
    }

    let mut ans = Box::new(ListNode::new(0));
    let mut ptr = &mut ans;
    let mut heap = BinaryHeap::new();
    // 把第一列的元素放到堆里。
    for node in lists {
        if let Some(n) = node {
            heap.push(n);
        }
    }
    // 弹出最小的，然后把它剩下的再加入到堆中。
    while let Some(mut node) = heap.pop() {
        if let Some(next) = node.next.take() {
            heap.push(next);
        }
        ptr.next = Some(node);
        ptr = ptr.next.as_mut().unwrap();
    }

    ans.next
}

#[cfg(feature = "all")]
fn reverse_k_group(head: Option<Box<ListNode>>, k: i32) -> Option<Box<ListNode>> {
    let mut remain = head;
    let mut dummy = Box::new(ListNode { val: 0, next: None });
    let mut tail = &mut dummy;
    while remain.is_some() {
        let (new_head, new_remain) = reverse_one(remain, k);
        remain = new_remain;
        tail.next = new_head;
        while tail.next.as_ref().is_some() {
            tail = tail.next.as_mut().unwrap();
        }
    }
    dummy.next
}

#[cfg(feature = "all")]
fn reverse_one(
    head: Option<Box<ListNode>>,
    k: i32,
) -> (Option<Box<ListNode>>, Option<Box<ListNode>>) {
    let mut pre = head.as_ref();
    for _ in 0..k {
        if pre.is_none() {
            return (head, None);
        }
        pre = pre.unwrap().next.as_ref();
    }

    let mut dummy = ListNode { val: 0, next: None };
    let mut remain = head;
    for _ in 0..k {
        if let Some(mut n) = remain {
            remain = n.next.take();
            n.next = dummy.next.take();
            dummy.next = Some(n);
        }
    }
    (dummy.next, remain)
}

#[cfg(feature = "all")]
fn remove_duplicates1(nums: &mut Vec<i32>) -> i32 {
    nums.dedup();
    nums.len() as i32
}

#[cfg(feature = "all")]
fn remove_duplicates2(nums: &mut Vec<i32>) -> i32 {
    let mut slow = 0;
    for fast in 0..nums.len() {
        if nums[slow] != nums[fast] {
            slow += 1;
            nums[slow] = nums[fast];
        }
    }
    (slow + 1) as i32
}

#[cfg(feature = "all")]
fn remove_element(nums: &mut Vec<i32>, val: i32) -> i32 {
    let mut slow = 0;
    for fast in 0..nums.len() {
        if nums[fast] != val {
            nums[slow] = nums[fast];
            slow += 1;
        }
    }
    (slow + 1) as i32
}

#[cfg(feature = "all")]
fn swap_pairs(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    head.and_then(|mut n| match n.next {
        Some(mut m) => {
            n.next = swap_pairs(m.next);
            m.next = Some(n);
            Some(m)
        }
        None => Some(n),
    })
}

#[cfg(feature = "all")]
fn reverse_k_group(head: Option<Box<ListNode>>, k: i32) -> Option<Box<ListNode>> {
    let length = {
        let mut ret = 0;
        let mut cur = &head;
        while cur.is_some() {
            cur = &cur.as_ref().unwrap().next;
            ret += 1;
        }
        ret
    };
    head
}

#[cfg(feature = "all")]
pub fn count_pairs(n: i32, edges: Vec<Vec<i32>>, mut queries: Vec<i32>) -> Vec<i32> {
    use std::cmp::Ordering;
    use std::collections::HashMap;

    assert!(n > 0);
    let n = n as usize; // 节点数目 > 0
    let mut degs: Vec<usize> = vec![0; n + 1]; // 节点的度
    let mut cnt_e: HashMap<usize, usize> = HashMap::new(); // 边的数量，(1, 2), (2, 1)视为2次， key = 1 * n + 2
    for edge in edges.iter() {
        let (x, y) = (edge[0], edge[1]);
        assert!(x >= 0);
        assert!(y >= 0);
        let (mut x, mut y) = (x as usize, y as usize);
        if x < y {
            (x, y) = (y, x);
        }
        degs[x] += 1;
        degs[y] += 1;
        let key = x * n + y;
        cnt_e.insert(key, cnt_e.get(&key).unwrap_or(&0) + 1);
    }

    let mut cnt_deg: HashMap<usize, usize> = HashMap::new(); // 度的数量，与python的Counter(degs)相同
    for &deg in degs[1..].iter() {
        cnt_deg.insert(deg, cnt_deg.get(&deg).unwrap_or(&0) + 1);
    }

    // 选择两个度，使得两数之和恰好等于 i 的方案数。
    let mut cnts = vec![0; degs.iter().max().unwrap_or(&0) * 2 + 2]; // 后缀和
    for (&deg1, &c1) in cnt_deg.iter() {
        for (&deg2, &c2) in cnt_deg.iter() {
            // 只考虑 deg1 > deg2 与 deg1 = deg2，避免重复
            match deg1.cmp(&deg2) {
                Ordering::Less => cnts[deg1 + deg2] += c1 * c2, // deg1, deg2不同，则乘法原理
                Ordering::Equal => cnts[deg1 + deg2] += c1 * (c1 - 1) >> 1, // deg1, deg2相同，则取组合数
                Ordering::Greater => {}
            }
        }
    }

    for (e, c) in cnt_e {
        // 点对 (x, y) 之间的连接不应计入 x与y 对外的连接数
        let (x, y) = (e / n, e % n);
        let s = degs[x] + degs[y];
        cnts[s] -= 1;
        cnts[s - c] += 1;
    }
    for i in (1..cnts.len()).rev() {
        cnts[i - 1] += cnts[i];
    }
    for q in queries.iter_mut() {
        *q = cnts[(*q as usize + 1).min(cnts.len() - 1)] as i32;
    }
    queries
}

#[cfg(feature = "all")]
fn remove_element(nums: &mut Vec<i32>, val: i32) -> i32 {
    let mut slow_idx = 0;
    for pos in 0..nums.len() {
        if nums[pos] != val {
            nums[slow_idx] = nums[pos];
            slow_idx += 1;
        }
    }
    slow_idx as i32
}

#[cfg(feature = "all")]
fn str_str(haystack: String, needle: String) -> i32 {
    match haystack.find(&needle) {
        Some(ret) => ret as i32,
        None => -1,
    }
}

#[cfg(feature = "all")]
fn find_substring(s: String, words: Vec<String>) -> Vec<i32> {
    use std::collections::HashMap;
    assert!(!words.is_empty() && !s.is_empty());
    let m = words.len();
    let n = words[0].len();
    let ans_len = m * n;
    let mut ret = vec![];
    if s.len() < ans_len {
        return ret;
    };

    let mut start = 0;
    while start < n {
        dbg!(start);
        let mut hm = HashMap::with_capacity(m);
        let mut left = start;
        while left + ans_len <= s.len() {
            match left {
                left if left < n => {
                    for i in 0..m {
                        let word_slice = &s[(left + i * n)..(left + (i + 1) * n)];
                        dbg!(word_slice);
                        hm.insert(word_slice, hm.get(word_slice).unwrap_or(&0) + 1);
                    }
                }
                _ => {
                    dbg!(left);
                    let word_slice = &s[(left + (m - 1) * n)..(left + ans_len)];
                    dbg!(word_slice);
                    hm.insert(word_slice, hm.get(word_slice).unwrap_or(&0) + 1);
                }
            }
            dbg!(&hm);
            let mut matched_word_num = 0;
            let mut hm_c = hm.clone();
            for w in words.iter() {
                match hm_c.get_mut(&w[..]) {
                    Some(x) if *x > 0 && matched_word_num < m => {
                        *x -= 1;
                        matched_word_num += 1;
                    }
                    _ => break,
                }
            }
            if matched_word_num == m {
                ret.push(left as i32);
            }
            let should_drop = &s[left..(left + n)];
            dbg!(should_drop);
            hm.insert(should_drop, hm.get(should_drop).unwrap() - 1);
            dbg!(&hm);
            left += n;
        }
        start += 1;
    }

    ret
}

#[cfg(feature = "all")]
fn find_substring_better(s: String, words: Vec<String>) -> Vec<i32> {
    use std::collections::HashMap;
    macro_rules! helper { // 哈希统计，为 0 时移除
        ($diff:expr, $s:expr, $cnt:expr) => {
            let t = $s as &str;
            *$diff.entry(t).or_insert(0) += $cnt;
            if *$diff.get(t).unwrap() == 0 {$diff.remove(t);}
        }
    }
    let mut diff = HashMap::new();
    let (m, n) = (words.len(), words[0].len());
    let mut ans = vec![];
    for idx in 0..n { // 仅需要分为 n 组
        if idx + m * n > s.len() {break}
        for i in (idx..idx + m * n).step_by(n) {
            helper!(diff, &s[i..i + n], 1);
        }
        for w in words.iter() {
            helper!(diff, w, -1);
        }
        if diff.is_empty() {ans.push(idx as i32)}
        for i in (idx + n..s.len() - m * n + 1).step_by(n) {
            helper!(diff, &s[i - n..i], -1); // 移除左边
            helper!(diff, &s[i + (m - 1) * n..i + m * n], 1); // 添加右边
            if diff.is_empty() {ans.push(i as i32)}
        }
        diff.clear();
    }
    ans
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test() {
    }
}
