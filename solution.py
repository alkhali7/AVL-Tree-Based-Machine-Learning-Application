"""
AVL-trees Project
CSE 331 FS23
Shams Al khalidy
solution.py
"""
import math
import queue
from typing import TypeVar, Generator, List, Tuple, Optional
from collections import deque
import json
from queue import SimpleQueue

# for more information on typehinting, check out https://docs.python.org/3/library/typing.html
T = TypeVar("T")  # represents generic type
# represents a Node object (forward-declare to use in Node __init__)
Node = TypeVar("Node")
# represents a custom type used in application
AVLWrappedDictionary = TypeVar("AVLWrappedDictionary")


class Node:
    """
    Implementation of an BST and AVL tree node.
    Do not modify.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["value", "parent", "left", "right", "height"]

    def __init__(self, value: T, parent: Node = None,
                 left: Node = None, right: Node = None) -> None:
        """
        Construct an AVL tree node.

        :param value: value held by the node object
        :param parent: ref to parent node of which this node is a child
        :param left: ref to left child node of this node
        :param right: ref to right child node of this node
        """
        self.value = value
        self.parent, self.left, self.right = parent, left, right
        self.height = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return f"<{str(self.value)}>"

    def __str__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return repr(self)


####################################################################################################

class BinarySearchTree:
    """
    Implementation of an BSTree.
    Modify only below indicated line.
    """

    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["origin", "size"]

    def __init__(self) -> None:
        """
        Construct an empty BST tree.
        """
        self.origin = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the BSTree as a string.

        :return: string representation of the BST tree
        """
        if self.origin is None:
            return "Empty BST Tree"

        lines = pretty_print_binary_tree(self.origin, 0, False, '-')[0]
        return "\n" + "\n".join((line.rstrip() for line in lines))

    def __str__(self) -> str:
        """
        Represent the BSTree as a string.

        :return: string representation of the BSTree
        """
        return repr(self)

    def visualize(self, filename="bst_visualization.svg"):
        """
        Generates an svg image file of the binary tree.

        :param filename: The filename for the generated svg file. Should end with .svg.
        Defaults to output.svg
        """
        svg_string = svg(self.origin, node_radius=20)
        print(svg_string, file=open(filename, 'w'))
        return svg_string

    def height(self, root: Node) -> int:
        """
        Calculates and returns the height of a subtree in the BSTree, handling the case where root is None.
         Note that an empty subtree has a height of -1.
        This method is simple to implement, particularly if you recall that a Node's height (if it exists) is stored in
         its height attribute.
        This function is not directly tested as it is very straightforward, but it will be utilized by other functions.
        Time / Space Complexity: O(1) / O(1).
        root: Node: The root of the subtree whose height is to be calculated.
        Returns: The height of the subtree at root, or -1 if root is None.
        """
        if root is None:
            return -1
        return root.height

    def insert(self, root: Node, val: T) -> None:
        """
        Inserts a node with the value val into the subtree rooted at root, returning the root of the balanced subtree
        after the insertion.
        Time / Space Complexity: O(h) / O(1), where h is the height of the tree.
        """
        if root is None:
            self.origin = Node(val)
            self.size += 1
        else:
            if root.value == val:
                return
            elif val < root.value:
                if root.left is None:
                    root.left = Node(val, parent=root)
                    self.size += 1
                else:
                    self.insert(root.left, val)
            elif val > root.value:
                if root.right is None:
                    root.right = Node(val, parent=root)
                    self.size += 1
                else:
                    self.insert(root.right, val)
            # recalculate height of tree after insertion
            root.height = max(root.left.height if root.left else -1, root.right.height if root.right else -1) + 1

    def remove(self, root: Node, val: T) -> Optional[Node]:
        """
        Removes the node with the value val from the subtree rooted at root, and returns the root of the subtree after
         the removal.
        If you are removing a node with two children, swap the value of this node with its predecessor, and then recursively remove the predecessor node (which will have the value to be removed after the swap and is guaranteed to be a leaf).
        Time / Space Complexity: O(h) / O(1), where h is the height of the tree. Returns: The root of the new subtree after the removal (could be the original root).
        """
        if root is None:
            return
        # search in the left subtree if val is lesser
        if val < root.value:
            root.left = self.remove(root.left, val)
        elif val > root.value:      # search in the right subtree
            root.right = self.remove(root.right, val)
        else:                   # val == root.value
            if root.left is None:
                self.size -= 1
                return root.right   # Node with only a right child, link the parent to the right child.
            elif root.right is None:
                self.size -= 1
                return root.left    # Node with only a left child, link the parent to the left child.
            # Node with two children, find predecessor, swap values, and delete the predecessor
            pred = root.left
            while pred.right is not None:
                pred = pred.right
            root.value = pred.value
            root.left = self.remove(root.left, pred.value)
            # Calculate and update the height of the current node after removal
        root.height = 1 + max(root.left.height if root.left else -1, root.right.height if root.right else -1)
        return root

    def search(self, root: Node, val: T) -> Optional[Node]:
        """
        Searches for and returns the Node containing the value val in the subtree rooted at root.
        If val is not present in the subtree, the function returns the Node below which val
        would be inserted as a child.
        For example, in a BST 1-2-3 tree (with 2 as the root and 1, 3 as children),
         search(node_2, 0) would return node_1 because the value 0 would be
         inserted as a left child of node_1.
        This method is simplest to implement recursively.
        Time / Space Complexity: O(h) / O(1), where h is the height of the tree.
        root: Node: The root of the subtree in which to search for val.
        val: T: The value to search for.
        Returns: The Node containing val, or the Node below which val would be inserted as a child if it does not exist.
        """
        # Tree is empty
        if root is None:
            return None
        # Search left subtree
        if val < root.value:
            # if no left subtree to search
            if root.left is None:
                return root     # return potential parent
            return self.search(root.left,val)
        elif val > root.value:
            if root.right is None:
                return root
            return self.search(root.right, val)
        else:       # node with value found
            return root


class AVLTree:
    """
    Implementation of an AVL tree.
    Modify only below indicated line.
    """

    __slots__ = ["origin", "size"]

    def __init__(self) -> None:
        """
        Construct an empty AVL tree.
        """
        self.origin = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree as a string.

        :return: string representation of the AVL tree
        """
        if self.origin is None:
            return "Empty AVL Tree"

        return super(AVLTree, self).__repr__()

    def __str__(self) -> str:
        """
        Represent the AVLTree as a string.

        :return: string representation of the BSTree
        """
        return repr(self)

    def visualize(self, filename="avl_tree_visualization.svg"):
        """
        Generates an svg image file of the binary tree.

        :param filename: The filename for the generated svg file. Should end with .svg.
        Defaults to output.svg
        """
        svg_string = svg(self.origin, node_radius=20)
        print(svg_string, file=open(filename, 'w'))
        return svg_string

    def height(self, root: Node) -> int:
        """
        Calculates the height of a subtree in the AVL tree, handling cases where root might be None. Remember, the height of an empty subtree is defined as -1.
        Parameters:
        root (Node): The root node of the subtree whose height you wish to determine.
        Returns: The height of the subtree rooted at root.
        Time / Space Complexity: O(1) / O(1)
        """
        if root is None:
            return -1
        return root.height

    def left_rotate(self, root: Node) -> Optional[Node]:
        """This method performs a left rotation on the subtree rooted at root,
        returning the new root of the subtree after the rotation.
        root (Node): The root node of the subtree that is to be rotated. Returns: The root of the new subtree post-rotation.
        Time / Space Complexity: O(1) / O(1) """
        if root is None:
            return root
        # no right child, cannot preform a left rotation
        if root.right is None:
            return root
        # set new root
        new_root = root.right
        # set new root to have same parent as prev root
        new_root.parent = root.parent
        if new_root.parent:
            if root == new_root.parent.left:
                new_root.parent.left = new_root
            else:
                new_root.parent.right = new_root
        # set old root to have its parent be new root
        root.parent = new_root
        left_subtree = new_root.left      # right children from old root.left.right

        new_root.left = root           # new origin gets old root as left  child
        root.right = left_subtree       # root gets the leftover right child from old right subtree
        if left_subtree:
            left_subtree.parent = root
        # update heights
        root.height = max(root.left.height if root.left else -1, root.right.height if root.right else -1) +1
        new_root.height = max(new_root.left.height if new_root.left else -1, new_root.right.height if new_root.right else -1) + 1
        # # update origin if we are rotating at origin
        if self.origin == root:
            self.origin = new_root
        return new_root

    def right_rotate(self, root: Node) -> Optional[Node]:
        """
        This method performs a right rotation on the subtree rooted at root, returning the new root of the subtree after the rotation.
        Parameters:
        Time / Space Complexity: O(1) / O(1)
        """
        if root is None:
            return root
        # no left child, cannot preform a right rotation
        if root.left is None:
            return root
        # set new root
        new_root = root.left
        # set new root to have same parent as prev root
        new_root.parent = root.parent
        if new_root.parent:
            if root == new_root.parent.left:
                new_root.parent.left = new_root
            else:
                new_root.parent.right = new_root
        # set old root to have its parent be new root
        root.parent = new_root
        right_subtree = new_root.right      # right children from old root.left.right

        new_root.right = root           # new origin gets right child as old root
        root.left = right_subtree       # root gets the leftover right child from old left subtree
        if right_subtree:           # update its parent to be old root
            right_subtree.parent = root

        # update heights
        root.height = max(root.left.height if root.left else -1, root.right.height if root.right else -1) +1
        new_root.height = max(new_root.left.height if new_root.left else -1, new_root.right.height if new_root.right else -1) + 1
        # # update origin if we are rotating at origin
        if self.origin == root:
            self.origin = new_root
        return new_root

    def balance_factor(self, root: Node) -> int:
        """
        This method computes the balance factor of the subtree rooted at root.
        The balance factor is calculated as h_L - h_R, where h_L is the height of the left subtree, and h_R is the
        height of the right subtree.
        In a properly balanced AVL tree, all nodes should have a balance factor in the set {-1, 0, +1}.
        A balance factor of -2 or +2 triggers a rebalance.
        For an empty subtree (where root is None), the balance factor is 0.
        To maintain time complexity, update the height attribute of each node during insertions/deletions/rebalances,
        allowing you to use h_L = left.height and h_R = right.height directly
        """
        # empty tree has a BF of 0
        if root is None:
            return 0
        return (root.left.height if root.left else -1) - (root.right.height if root.right else -1)

    def rebalance(self, root: Node) -> Optional[Node]:
        """
        This function rebalances the subtree rooted at root if it is unbalanced, and returns the root of the resulting
        subtree post-rebalancing.
        A subtree is considered unbalanced if the balance factor b of the root satisfies b >= 2 or b <= -2.
        There are four types of imbalances possible in an AVL tree, each requiring a specific sequence of rotations to
        restore balance. You can find more details on these here.
        Parameters:
        root (Node): The root of the subtree that potentially needs rebalancing.
        Returns: The root of the new, potentially rebalanced subtree.
        Time / Space Complexity: O(1) / O(1)
        """
        if root is None:
            return
        BF = self.balance_factor(root)

        # subtree needs left rotation
        if BF <= -2:
            # check if its a double rotation and needs a right rotation first, -2,1 case
            if self.balance_factor(root.right) >= 1:
                root.right = self.right_rotate(root.right)
            return self.left_rotate(root)
        # right rotation BF=2
        if BF >=2:
            if self.balance_factor(root.left) <= -1:  # 2,-1 case
                root.left = self.left_rotate(root.left)
            return self.right_rotate(root)
        return root

    def insert(self, root: Node, val: T) -> Optional[Node]:
        """This function inserts a new node with value val into the subtree rooted at root, balancing the subtree as necessary,  and returns the root of the resulting subtree.
        If a node with value val already exists in the tree, the function does nothing. This function updates the size and origin attributes of the AVLTree, sets the parent/child pointers
        correctly when inserting the new Node, updates the height attribute of affected nodes, and calls rebalance on all affected ancestor nodes.
        Time / Space Complexity: O(log n) / O(1)"""
        if root is None:
            self.origin = Node(val)
            self.size += 1
            return self.origin
        else:
            if root.value == val:
                return root
            elif val < root.value:
                if root.left is None:
                    new_node = Node(val, parent=root)
                    root.left = new_node
                    self.size += 1
                else:
                    self.insert(root.left, val)
            elif val > root.value:
                if root.right is None:
                    new_node = Node(val, parent=root)
                    root.right = new_node
                    self.size += 1
                else:
                    self.insert(root.right, val)
            # recalculate height of tree after insertion
            root.height = max(root.left.height if root.left else -1, root.right.height if root.right else -1) + 1
            new_root = self.rebalance(root)         # rebalance the tree
            return new_root

    def remove(self, root: Node, val: T) -> Optional[Node]:
        """
        This function removes the node with value val from the subtree rooted at root, balances the subtree as necessary
        , and returns the root of the resulting subtree.
        The function is implemented recursively.
        Parameters:
        root (Node): The root of the subtree from which val is to be removed.
        val (T): The value to be removed.
        Returns: The root of the new, balanced subtree.
        Time / Space Complexity: O(log n) / O(1)
        """
        if root is None:    # empty tree
            return
        # search in the left subtree if val is lesser
        if val < root.value:
            root.left = self.remove(root.left, val)
        elif val > root.value:      # search in the right subtree
            root.right = self.remove(root.right, val)
        else:                   # val == root.value
            if root.left is None:
                self.size -= 1
                return root.right   # Node with only a right child, link the parent to the right child.
            elif root.right is None:
                self.size -= 1
                return root.left    # Node with only a left child, link the parent to the left child.

            # Node with two children, find predecessor, swap values, and delete the predecessor
            pred = root.left
            while pred.right is not None:
                pred = pred.right
            root.value = pred.value
            root.left = self.remove(root.left, pred.value)

            # Calculate and update the height of the current node after removal
        root.height = 1 + max(root.left.height if root.left else -1, root.right.height if root.right else -1)
        # only new thing is the rebalance
        new_root = self.rebalance(root)
        return new_root

    def min(self, root: Node) -> Optional[Node]:
        """
        This function searches for and returns the Node containing the smallest value within the subtree rooted at root.
        The implementation of this function is most straightforward when done recursively.
        Parameters:
        root (Node): The root of the subtree within which to search for the minimum value.
        Returns: A Node object that holds the smallest value in the subtree rooted at root.
        Time / Space Complexity: O(log n) / O(1)
        """
        # empty tree
        if root is None:
            return None
        # go left as long as there is a min value (left subtree)
        if root.left is None:
            return root
        return self.min(root.left)

    def max(self, root: Node) -> Optional[Node]:
        """
        This function searches for and returns the Node containing the largest value within the subtree rooted at root.
        Like the min function, the implementation of this function is most straightforward when done recursively.
        Parameters:
        root (Node): The root of the subtree within which to search for the maximum value.
        Returns: A Node object that holds the largest value in the subtree rooted at root.
        Time / Space Complexity: O(log n) / O(1)
        """
        if root is None:
            return
        if root.right is None:
            return root
        return self.max(root.right)

    def search(self, root: Node, val: T) -> Optional[Node]:
        """
        This function searches for the Node with the value val within the subtree rooted at root.
        Returns: A Node object containing val if it exists within the subtree, and if not, the Node under which
        val would be inserted as a child.
        Time / Space Complexity: O(log n) / O(1)
        """
        if root is None:
            return
        if val < root.value:
            if root.left is None:
                return root
            return self.search(root.left, val)
        if val > root.value:
            if root.right is None:
                return root
            return self.search(root.right, val)
        else:
            return root

    def inorder(self, root: Node) -> Generator[Node, None, None]:
        """
        This function performs an inorder traversal (left, current, right) of the subtree rooted at root, generating the
         nodes one at a time using a Python generator.
        Use yield to produce individual nodes as they are encountered, and yield from for recursive calls to inorder.
        Ensure that None-type nodes are not yielded.
        Important: To pass the test case for this function, you must also make the AVLTree class iterable, enabling the
         usage of for node in avltree to iterate over the tree in an inorder manner.
        Time / Space Complexity: O(n) / O(1). Although the entire tree is traversed, the generator yields nodes one at a time, resulting in constant space complexity.
        Parameters:
        root (Node): The root node of the current subtree being traversed.
        Returns: A generator yielding the nodes of the subtree in inorder.
        """
        if root is None:
            return None
        if root:
            yield from self.inorder(root.left)    # allows me to yield from root left
            yield root
            yield from self.inorder(root.right)

    def __iter__(self) -> Generator[Node, None, None]:
        """
        This method makes the AVL tree class iterable, allowing you to use it in loops like for node in tree.
        For the iteration to work, this function should be implemented such that it returns the generator from the inorder traversal of the tree.
        Returns: A generator yielding the nodes of the tree in inorder.
        Implementation Note: This function should be one line, calling the inorder function.
        """
        return self.inorder(self.origin)

    def preorder(self, root: Node) -> Generator[Node, None, None]:
        """
        This function performs a preorder traversal (current, left, right) of the subtree rooted at root, generating the nodes one at a time using a Python generator.
        Use yield to produce individual nodes as they are encountered, and yield from for recursive calls to preorder.
        Ensure that None-type nodes are not yielded.
        Time / Space Complexity: O(n) / O(1). Although the entire tree is traversed, the generator yields nodes one at a time, resulting in constant space complexity.
        """
        if root is None:
            return None
        yield root
        yield from self.preorder(root.left)
        yield from self.preorder(root.right)

    def postorder(self, root: Node) -> Generator[Node, None, None]:
        """
        This function performs a postorder traversal (left, right, current) of the subtree rooted at root, generating the nodes one at a time using a Python generator.
        Utilize yield to produce individual nodes as they are encountered, and yield from for recursive calls to postorder.
        Ensure that None-type nodes are not yielded.
        Time / Space Complexity: O(n) / O(1). The entire tree is traversed, but the use of a generator yields nodes one at a time, maintaining constant space complexity.
        Parameters:
        root (Node): The root node of the current subtree being traversed.
        Returns: A generator yielding the nodes of the subtree in postorder. A StopIteration exception is raised once all nodes have been yielded.
        """
        if root is None:
            return None
        yield from self.postorder(root.left)
        yield from self.postorder(root.right)
        yield root

    def levelorder(self, root: Node) -> Generator[Node, None, None]:
        """
        This function performs a level-order (breadth-first) traversal of the subtree rooted at root, generating the nodes one at a time using a Python generator.
        Use the queue.SimpleQueue class for maintaining the queue of nodes during the traversal. Refer to the official documentation for more information.
        Utilize yield to produce individual nodes as they are encountered.
        Ensure that None-type nodes are not yielded.
        Time / Space Complexity: O(n) / O(n). The entire tree is traversed, and due to the nature of level-order traversal, the queue can grow to O(n), particularly in a perfect binary tree scenario.
        Parameters:
        root (Node): The root node of the current subtree being traversed.
        Returns: A generator yielding the nodes of the subtree in level-order. A StopIteration exception is raised once all nodes have been yielded.
        """
        if root is None:
            return None
        # create a queue for the level order traversal
        q = SimpleQueue()
        # Enqueue the root node
        q.put(root)
        while not q.empty():
            cuurent = q.get()
            # yield the current node
            yield cuurent
            # Enqueue left and right children if they exist
            if cuurent.left:
                q.put(cuurent.left)
            if cuurent.right:
                q.put(cuurent.right)

####################################################################################################

class AVLWrappedDictionary:
    """
    Implementation of a helper class which will be used as tree node values in the
    NearestNeighborClassifier implementation.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["key", "dictionary"]

    def __init__(self, key: float) -> None:
        """
        Construct a AVLWrappedDictionary with a key to search/sort on and a dictionary to hold data.

        :param key: floating point key to be looked up by.
        """
        self.key = key
        self.dictionary = {}

    def __repr__(self) -> str:
        """
        Represent the AVLWrappedDictionary as a string.

        :return: string representation of the AVLWrappedDictionary.
        """
        pprinted_dict = json.dumps(self.dictionary, indent=2)
        return f"key: {self.key} dict:{self.dictionary}"

    def __str__(self) -> str:
        """
        Represent the AVLWrappedDictionary as a string.

        :return: string representation of the AVLWrappedDictionary.
        """
        return repr(self)

    def __eq__(self, other: AVLWrappedDictionary) -> bool:
        """
        Implement == operator to compare 2 AVLWrappedDictionaries by key only.

        :param other: other AVLWrappedDictionary to compare with
        :return: boolean indicating whether keys of AVLWrappedDictionaries are equal
        """
        return abs(self.key - other.key) < 1e-6

    def __lt__(self, other: AVLWrappedDictionary) -> bool:
        """
        Implement < operator to compare 2 AVLWrappedDictionarys by key only.

        :param other: other AVLWrappedDictionary to compare with
        :return: boolean indicating ordering of AVLWrappedDictionaries
        """
        return self.key < other.key and not abs(self.key - other.key) < 1e-6

    def __gt__(self, other: AVLWrappedDictionary) -> bool:
        """
        Implement > operator to compare 2 AVLWrappedDictionaries by key only.

        :param other: other AVLWrappedDictionary to compare with
        :return: boolean indicating ordering of AVLWrappedDictionaries
        """
        return self.key > other.key and not abs(self.key - other.key) < 1e-6


class NearestNeighborClassifier:
    """
    Implementation of a one-dimensional nearest-neighbor classifier with AVL tree lookups.
    Modify only below indicated line.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["resolution", "tree"]

    def __init__(self, resolution: int) -> None:
        """
        Construct a one-dimensional nearest neighbor classifier with AVL tree lookups.
        Data are assumed to be floating point values in the closed interval [0, 1].

        :param resolution: number of decimal places the data will be rounded to, effectively
                           governing the capacity of the model - for example, with a resolution of
                           1, the classifier could maintain up to 11 nodes, spaced 0.1 apart - with
                           a resolution of 2, the classifier could maintain 101 nodes, spaced 0.01
                           apart, and so on - the maximum number of nodes is bounded by
                           10^(resolution) + 1.
        """
        self.tree = AVLTree()
        self.resolution = resolution

        # pre-construct lookup tree with AVLWrappedDictionary objects storing (key, dictionary)
        # pairs, but which compare with <, >, == on key only
        for i in range(10 ** resolution + 1):
            w_dict = AVLWrappedDictionary(key=(i / 10 ** resolution))
            self.tree.insert(self.tree.origin, w_dict)

    def __repr__(self) -> str:
        """
        Represent the NearestNeighborClassifier as a string.
        :return: string representation of the NearestNeighborClassifier.
        """
        return f"NNC(resolution={self.resolution}):\n{self.tree}"

    def __str__(self) -> str:
        """
        Represent the NearestNeighborClassifier as a string.
        :return: string representation of the NearestNeighborClassifier.
        """
        return repr(self)

    def visualize(self, filename: str = "nnc_visualization.svg") -> str:
        svg_string = svg(self.tree.origin, 48, nnc_mode=True)
        print(svg_string, file=open(filename, 'w'))
        return svg_string

    ########################################

    def fit(self, data: List[Tuple[float, str]]) -> None:
        """
        fit(self, data: List[Tuple[float, str]]) -> None
        The fit method is used to train the classifier with a dataset, helping it
        learn the associations between features x (float values) and target labels y (string values).
        Time: O(n log n)
        Space: O(n)
        Parameters:
        data (List[Tuple[float, str]]): A list of (x, y) pairs.
        Returns:
        None
        """
        # iterate through each (x, y) pair in the dataset
        for x, y in data:
            # round the x value to the specified precision (based on self.resolution)
            x_rounded = round(x, self.resolution)
            # # Temporarily change the type of x_rounded to an AVLWrappedDictionary for searching
            # Find the corresponding node in the tree using the rounded x value as a key
            node = self.tree.search(self.tree.origin, AVLWrappedDictionary(key=x_rounded))
            # print(AVLWrappedDictionary(key=x_rounded))

            # access the AVLWrappedDictionary object at this node
            avl_dict = node.value
            # update the count of the label y in this dictionary
            if y in avl_dict.dictionary:
                avl_dict.dictionary[y] += 1
            else:
                avl_dict.dictionary[y] = 1

    def predict(self, x: float, delta: float) -> Optional[str]:
        """
        The predict method predicts the target label y for a given feature value x.
        Process:
        Round the x value to the specified precision (based on self.resolution).
        Search for nodes in the tree whose keys are within ± delta of the rounded x value.
        Access the AVLWrappedDictionary objects at these nodes.
        Determine the most common label y across all these dictionaries.
        The method effectively predicts y based on the most common y values observed in the training data for x values close to the input.
        If there are no data points close to the input, the function returns None.
        Complexity:
        Time: O(k log n)
        Space: O(1)
        Parameters:
        x (float): Feature value to be predicted.
        delta (float): Range to search for neighboring feature values.
        Returns:
        A string (str) representing the predicted target label y.
        """
        # Round the x value to the specified precision
        x_rounded = round(x, self.resolution)

        # Calculate the lower and upper bounds of the range
        lower_bound = x_rounded - delta
        upper_bound = x_rounded + delta

        # Create a dictionary to store label counts
        label_counts = {}

        # Perform searches for keys within the specified range
        for key in range(int(lower_bound * 10**self.resolution), int(upper_bound * 10**self.resolution) + 1):
            # Create an AVLWrappedDictionary object with the key for searching
            key_to_search_avl = AVLWrappedDictionary(key=key / 10**self.resolution)

            # Search for the key in the tree
            node = self.tree.search(self.tree.origin, key_to_search_avl)
            if node:
                avl_dict = node.value
                for label, count in avl_dict.dictionary.items():
                    if label in label_counts:
                        label_counts[label] += count
                    else:
                        label_counts[label] = count

        # Find the label with the highest count
        most_common_label = None
        max_count = 0
        for label, count in label_counts.items():
            if count > max_count:
                most_common_label = label
                max_count = count

        # Return the most common label, or None if there are no data points close to the input
        return most_common_label


####################################################################################################


"""
For the curious people, the following functions are used to visualize and compare the performance
of BinarySearchTree and AVLTree under extreme conditions. You do not need to modify (or run) these 
functions, but you are welcome to play around with them if you wish. 

You should know that AVLTree is faster than BinarySearchTree in the worst case, but how much faster?
The following functions will help you answer this question, shall you choose to try them out.

Uncomment the line under "if __name__ == '__main__':" to run the performance comparison (after completing
the rest of the project) and you will be greeted with a plot of the results. The function will require
matplotlib, make sure to install it if you do not have it already.
"""


def compare_times(structure: dict, sizes: List[int], trial: int) -> dict:
    """
    Comparing time on provide data structures in the worst case of BST tree
    :param structure: provided data structures
    :param sizes: size of test input
    :param trial: number of trials to test
    :return: dict with list of average times per input size for each algorithm
    """
    import sys
    import time
    result = {}
    sys.stdout.write('\r')
    sys.stdout.write('Start...\n')
    total = len(sizes) * len(structure)
    count = 0
    for algorithm, value in structure.items():
        ImplementedTree = value
        if algorithm not in result:
            result[algorithm] = []
        for size in sizes:
            sum_times = 0
            for _ in range(trial):
                tree = ImplementedTree()
                start = time.perf_counter()
                for i in range(size):
                    tree.insert(tree.origin, i)
                for i in range(size, -1, -1):
                    tree.remove(tree.origin, i)
                end = time.perf_counter()
                sum_times += (end - start)
            count += 1
            result[algorithm].append(sum_times / trial)
            sys.stdout.write("[{:<20s}] {:d}%\n".format('=' * ((count * 20) // total),
                                                        count * 100 // total))
            sys.stdout.flush()
    return result


def plot_time_comparison():
    """
    Use compare_times to make a time comparison of normal binary search tree and AVL tree
    in a worst case scenario.
    Requires matplotlib. Comment this out if you do not wish to install matplotlib.
    """
    import matplotlib.pyplot as plt
    import sys
    sys.setrecursionlimit(2010)
    structures = {
        "bst": BinarySearchTree,
        "avl": AVLTree
    }
    sizes = [4, 5, 6, 7, 8, 9, 10, 25, 50, 100, 300, 500, 1000, 2000]
    trials = 5
    data = compare_times(structures, sizes, trials)

    plt.style.use('seaborn-colorblind')
    plt.figure(figsize=(12, 8))

    for structure in structures:
        plt.plot(sizes, data[structure], label=structure)
    plt.legend()
    plt.xlabel("Input Size")
    plt.ylabel("Time to Sort (sec)")
    plt.title("BST vs AVL")
    plt.show()


_SVG_XML_TEMPLATE = """
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
<style>
    .value {{
        font: 300 16px monospace;
        text-align: center;
        dominant-baseline: middle;
        text-anchor: middle;
    }}
    .dict {{
        font: 300 16px monospace;
        dominant-baseline: middle;
    }}
    .node {{
        fill: lightgray;
        stroke-width: 1;
    }}
</style>
<g stroke="#000000">
{body}
</g>
</svg>
"""

_NNC_DICT_BOX_TEXT_TEMPLATE = """<text class="dict" y="{y}" xml:space="preserve">
    <tspan x="{label_x}" dy="1.2em">{label}</tspan>
    <tspan x="{bracket_x}" dy="1.2em">{{</tspan>
    {values}
    <tspan x="{bracket_x}" dy="1.2em">}}</tspan>
</text>
"""


def pretty_print_binary_tree(root: Node, curr_index: int, include_index: bool = False,
                             delimiter: str = "-", ) -> \
        Tuple[List[str], int, int, int]:
    """
    Taken from: https://github.com/joowani/binarytree

    Recursively walk down the binary tree and build a pretty-print string.
    In each recursive call, a "box" of characters visually representing the
    current (sub)tree is constructed line by line. Each line is padded with
    whitespaces to ensure all lines in the box have the same length. Then the
    box, its width, and start-end positions of its root node value repr string
    (required for drawing branches) are sent up to the parent call. The parent
    call then combines its left and right sub-boxes to build a larger box etc.
    :param root: Root node of the binary tree.
    :type root: binarytree.Node | None
    :param curr_index: Level-order_ index of the current node (root node is 0).
    :type curr_index: int
    :param include_index: If set to True, include the level-order_ node indexes using
        the following format: ``{index}{delimiter}{value}`` (default: False).
    :type include_index: bool
    :param delimiter: Delimiter character between the node index and the node
        value (default: '-').
    :type delimiter:
    :return: Box of characters visually representing the current subtree, width
        of the box, and start-end positions of the repr string of the new root
        node value.
    :rtype: ([str], int, int, int)
    .. _Level-order:
        https://en.wikipedia.org/wiki/Tree_traversal#Breadth-first_search
    """
    if root is None:
        return [], 0, 0, 0

    line1 = []
    line2 = []
    if include_index:
        node_repr = "{}{}{}".format(curr_index, delimiter, root.value)
    else:
        if type(root.value) == AVLWrappedDictionary:
            node_repr = f'{root.value},h={root.height},' \
                        f'⬆{str(root.parent.value.key) if root.parent else "None"}'
        else:
            node_repr = f'{root.value},h={root.height},' \
                        f'⬆{str(root.parent.value) if root.parent else "None"}'

    new_root_width = gap_size = len(node_repr)

    # Get the left and right sub-boxes, their widths, and root repr positions
    l_box, l_box_width, l_root_start, l_root_end = pretty_print_binary_tree(
        root.left, 2 * curr_index + 1, include_index, delimiter
    )
    r_box, r_box_width, r_root_start, r_root_end = pretty_print_binary_tree(
        root.right, 2 * curr_index + 2, include_index, delimiter
    )

    # Draw the branch connecting the current root node to the left sub-box
    # Pad the line with whitespaces where necessary
    if l_box_width > 0:
        l_root = (l_root_start + l_root_end) // 2 + 1
        line1.append(" " * (l_root + 1))
        line1.append("_" * (l_box_width - l_root))
        line2.append(" " * l_root + "/")
        line2.append(" " * (l_box_width - l_root))
        new_root_start = l_box_width + 1
        gap_size += 1
    else:
        new_root_start = 0

    # Draw the representation of the current root node
    line1.append(node_repr)
    line2.append(" " * new_root_width)

    # Draw the branch connecting the current root node to the right sub-box
    # Pad the line with whitespaces where necessary
    if r_box_width > 0:
        r_root = (r_root_start + r_root_end) // 2
        line1.append("_" * r_root)
        line1.append(" " * (r_box_width - r_root + 1))
        line2.append(" " * r_root + "\\")
        line2.append(" " * (r_box_width - r_root))
        gap_size += 1
    new_root_end = new_root_start + new_root_width - 1

    # Combine the left and right sub-boxes with the branches drawn above
    gap = " " * gap_size
    new_box = ["".join(line1), "".join(line2)]
    for i in range(max(len(l_box), len(r_box))):
        l_line = l_box[i] if i < len(l_box) else " " * l_box_width
        r_line = r_box[i] if i < len(r_box) else " " * r_box_width
        new_box.append(l_line + gap + r_line)

    # Return the new box, its width and its root repr positions
    return new_box, len(new_box[0]), new_root_start, new_root_end


def svg(root: Node, node_radius: int = 16, nnc_mode=False) -> str:
    """
    Taken from: https://github.com/joowani/binarytree

    Generate SVG XML.
    :param root: Generate SVG for tree rooted at root
    :param node_radius: Node radius in pixels (default: 16).
    :type node_radius: int
    :return: Raw SVG XML.
    :rtype: str
    """
    tree_height = root.height
    scale = node_radius * 3
    xml = deque()
    nodes_for_nnc_visualization: list[AVLWrappedDictionary] = []

    def scale_x(x: int, y: int) -> float:
        diff = tree_height - y
        x = 2 ** (diff + 1) * x + 2 ** diff - 1
        return 1 + node_radius + scale * x / 2

    def scale_y(y: int) -> float:
        return scale * (1 + y)

    def add_edge(parent_x: int, parent_y: int, node_x: int, node_y: int) -> None:
        xml.appendleft(
            '<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>'.format(
                x1=scale_x(parent_x, parent_y),
                y1=scale_y(parent_y),
                x2=scale_x(node_x, node_y),
                y2=scale_y(node_y),
            )
        )

    def add_node(node_x: int, node_y: int, node: Node) -> None:
        x, y = scale_x(node_x, node_y), scale_y(node_y)
        xml.append(
            f'<circle class="node" cx="{x}" cy="{y}" r="{node_radius}"/>')

        if nnc_mode:
            nodes_for_nnc_visualization.append(node.value)
            xml.append(
                f'<text class="value" x="{x}" y="{y + 5}">key={node.value.key}</text>')
        else:
            xml.append(
                f'<text class="value" x="{x}" y="{y + 5}">{node.value}</text>')

    current_nodes = [root.left, root.right]
    has_more_nodes = True
    y = 1

    add_node(0, 0, root)

    while has_more_nodes:

        has_more_nodes = False
        next_nodes: List[Node] = []

        for x, node in enumerate(current_nodes):
            if node is None:
                next_nodes.append(None)
                next_nodes.append(None)
            else:
                if node.left is not None or node.right is not None:
                    has_more_nodes = True

                add_edge(x // 2, y - 1, x, y)
                add_node(x, y, node)

                next_nodes.append(node.left)
                next_nodes.append(node.right)

        current_nodes = next_nodes
        y += 1

    svg_width = scale * (2 ** tree_height)
    svg_height = scale * (2 + tree_height)
    if nnc_mode:

        line_height = 20
        box_spacing = 10
        box_margin = 5
        character_width = 10

        max_key_count = max(
            map(lambda obj: len(obj.dictionary), nodes_for_nnc_visualization))
        box_height = (max_key_count + 3) * line_height + box_margin

        def max_length_item_of_node_dict(node: AVLWrappedDictionary):
            # Check if dict is empty so max doesn't throw exception
            if len(node.dictionary) > 0:
                item_lengths = map(lambda pair: len(
                    str(pair)), node.dictionary.items())
                return max(item_lengths)
            return 0

        max_value_length = max(
            map(max_length_item_of_node_dict, nodes_for_nnc_visualization))
        box_width = max(max_value_length * character_width, 110)

        boxes_per_row = svg_width // box_width
        rows_needed = math.ceil(
            len(nodes_for_nnc_visualization) / boxes_per_row)

        nodes_for_nnc_visualization.sort(key=lambda node: node.key)
        for index, node in enumerate(nodes_for_nnc_visualization):
            curr_row = index // boxes_per_row
            curr_column = index % boxes_per_row

            box_x = curr_column * (box_width + box_spacing)
            box_y = curr_row * (box_height + box_spacing) + svg_height
            box = f'<rect x="{box_x}" y="{box_y}" width="{box_width}" ' \
                  f'height="{box_height}" fill="white" />'
            xml.append(box)

            value_template = '<tspan x="{value_x}" dy="1.2em">{key}: {value}</tspan>'
            text_x = box_x + 10

            def item_pair_to_svg(pair):
                return value_template.format(key=pair[0], value=pair[1], value_x=text_x + 10)

            values = map(item_pair_to_svg, node.dictionary.items())
            text = _NNC_DICT_BOX_TEXT_TEMPLATE.format(
                y=box_y,
                label=f"key = {node.key}",
                label_x=text_x,
                bracket_x=text_x,
                values='\n'.join(values)
            )
            xml.append(text)

        svg_width = boxes_per_row * (box_width + box_spacing * 2)
        svg_height += rows_needed * (box_height + box_spacing * 2)

    return _SVG_XML_TEMPLATE.format(
        width=svg_width,
        height=svg_height,
        body="\n".join(xml),
    )


if __name__ == "__main__":
    # plot_time_comparison()
    pass
