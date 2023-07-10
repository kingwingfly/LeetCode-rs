#![allow(unused)]

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

#[cfg(target_feature = "all")]
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

fn int_to_roman(num: i32) -> String {}

fn main() {
    let v = max_area(vec![1, 8, 6, 2, 5, 4, 8, 3, 7]);
    dbg!(v);
}
